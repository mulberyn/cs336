import json
import mmap
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import regex as re
from tqdm import tqdm


class Word:
    """Represents a word as a sequence of token IDs with frequency and local pair-count statistics.

    Each ``Word`` tracks the token sequence (as integer IDs), the word's
    frequency in the corpus, and the counts of every adjacent token pair
    that appears inside the word.  These local pair counts are used during
    incremental BPE training to efficiently update global pair statistics
    after a merge.

    Attributes:
        tokens: List of integer token IDs that make up this word.
        freq: Total occurrence count of this word in the training corpus.
        pair_counts: Counter mapping each adjacent token pair ``(a, b)``
            to how many times it appears within this specific word.
    """

    __slots__ = ("tokens", "freq", "pair_counts")

    tokens: list[int]
    freq: int
    pair_counts: Counter[tuple[int, int]]

    def __init__(self, tokens: list[int], freq: int) -> None:
        """Initialize a Word with its token sequence and corpus frequency.

        Args:
            tokens: List of integer token IDs for this word.
            freq: Number of times this word appears in the training corpus.
        """
        self.tokens = tokens
        self.freq = freq
        self.pair_counts = Counter()
        for i in range(len(tokens) - 1):
            self.pair_counts[(tokens[i], tokens[i + 1])] += 1


def build_structures(
    token_sequences: dict[bytes, int],
) -> tuple[dict[int, Word], dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
    """Build the incremental-training data structures from pre-tokenized sequences.

    Args:
        token_sequences: Mapping from a **bytes** key (a UTF-8 encoded word) to
            its total frequency in the training corpus.  Each byte in the key
            is treated as a single-byte token ID.

    Returns:
        A tuple of ``(words, pair_counts, pair_to_words)`` where:

        * **words** -- ``{word_id: Word}`` mapping for every unique sequence.
        * **pair_counts** -- ``{(a, b): global_freq}`` for every adjacent pair
          across all words.
        * **pair_to_words** -- ``{(a, b): {word_id, ...}}`` inverted index
          that records which words contain a given pair.
    """
    words: dict[int, Word] = {}
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    for word_id, (seq, freq) in enumerate(token_sequences.items()):
        tokens = list(seq)
        words[word_id] = Word(tokens, freq)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_id)

    return words, pair_counts, pair_to_words


class ReverseBytes:
    """Thin wrapper that inverts byte-string comparison so Python's min-heap
    behaves like a max-heap with respect to the byte representation of tokens.

    When Python compares two ``ReverseBytes`` objects it delegates to the
    wrapped ``data`` tuple but *reverses* the ``<`` operator.  This makes
    lexicographically *larger* byte strings sort *before* smaller ones inside
    ``heapq``, which is exactly the tie-breaking rule required by GPT-2 BPE.
    """

    __slots__ = ("data",)

    def __init__(self, data: tuple[bytes, bytes]) -> None:
        self.data = data

    def __lt__(self, other: "ReverseBytes") -> bool:
        return self.data > other.data


import heapq


def _make_heap_item(
    count: int,
    pair: tuple[int, int],
    vocab: dict[int, bytes],
) -> tuple[int, ReverseBytes, int, int]:
    """Build a single heap entry for the pair ``(a, b)``.

    The sort key is ``(-count, ReverseBytes(vocab[a]), ReverseBytes(vocab[b]))``
    so that the most frequent pair surfaces first, with byte-lexicographic
    tie-breaking.
    """
    a, b = pair
    return (-count, ReverseBytes((vocab[a], vocab[b])), a, b)


def init_heap(
    pair_counts: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
) -> list:
    """Build and heapify the initial max-heap from the full pair-count table.

    Args:
        pair_counts: Global ``{(a, b): count}`` mapping.
        vocab: Current vocabulary ``{id: bytes}``.

    Returns:
        A heapified list of ``(-count, ReverseBytes, a, b)`` tuples.
    """
    heap = [
        _make_heap_item(count, pair, vocab)
        for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)
    return heap


def push_heap(
    heap: list,
    pair: tuple[int, int],
    new_count: int,
    vocab: dict[int, bytes],
) -> None:
    """Push an updated ``(pair, new_count)`` entry onto the heap.

    Stale entries (those whose count no longer matches the current
    ``pair_counts``) are silently skipped inside :func:`pop_best_pair`.
    """
    heapq.heappush(heap, _make_heap_item(new_count, pair, vocab))

from typing import Optional

def pop_best_pair(
    heap: list,
    pair_counts: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
) -> Optional[tuple[int, int]]:
    """Pop the highest-priority pair from the heap, skipping stale entries.

    A heap entry is **stale** when its stored ``-count`` no longer matches
    the current value in ``pair_counts``.  Stale entries are discarded
    automatically.

    Args:
        heap: The min-heap of ``(-count, ReverseBytes, a, b)`` entries.
        pair_counts: Current global ``{(a, b): count}`` mapping.
        vocab: Current vocabulary (unused; accepted for interface uniformity).

    Returns:
        The best ``(a, b)`` pair, or ``None`` when the heap is exhausted.
    """
    while heap:
        neg_count, _reversed_key, a, b = heapq.heappop(heap)
        pair = (a, b)
        current = pair_counts.get(pair, 0)
        if current == -neg_count:
            return pair
    return None


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def update_word(
    word: Word,
    a: int,
    b: int,
    new_id: int,
    pair_counts: dict[tuple[int, int], int],
    pair_to_words: dict[tuple[int, int], set[int]],
    word_id: int,
    heap: Optional[list] = None,
    vocab: Optional[dict[int, bytes]] = None,
) -> None:
    """Replace every occurrence of the pair ``(a, b)`` with ``new_id`` inside
    *word*, and propagate the resulting pair-count deltas to the global
    ``pair_counts`` and ``pair_to_words`` structures.

    When *heap* and *vocab* are both provided, updated pair counts are also
    pushed onto the heap so that later calls to :func:`pop_best_pair` see the
    freshest counts.

    Args:
        word: The ``Word`` being updated in-place.
        a: First token ID of the pair to replace.
        b: Second token ID of the pair to replace.
        new_id: Token ID that replaces every ``(a, b)`` occurrence.
        pair_counts: Global ``{(a, b): count}`` mapping (mutated in-place).
        pair_to_words: Global ``{(a, b): {word_id, ...}}`` inverted index
            (mutated in-place).
        word_id: The integer ID of *word* (used for inverted-index updates).
        heap: Optional heap; when provided, every modified pair count is
            pushed as a fresh entry.
        vocab: Required when *heap* is provided; used to build heap entries.
    """
    old_tokens = word.tokens
    freq = word.freq

    # --- 1. Build the new token sequence after merging (a, b) → new_id ---
    new_tokens: list[int] = []
    i = 0
    while i < len(old_tokens):
        if i < len(old_tokens) - 1 and old_tokens[i] == a and old_tokens[i + 1] == b:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(old_tokens[i])
            i += 1

    # --- 2. Compute new local pair counts ---
    new_local: Counter[tuple[int, int]] = Counter()
    for i in range(len(new_tokens) - 1):
        new_local[(new_tokens[i], new_tokens[i + 1])] += 1

    old_local = word.pair_counts

    # --- 3. Compute per-pair delta between new and old local counts ---
    all_pairs = set(old_local.keys()) | set(new_local.keys())
    delta_local: dict[tuple[int, int], int] = {}
    for pair in all_pairs:
        delta = new_local.get(pair, 0) - old_local.get(pair, 0)
        if delta != 0:
            delta_local[pair] = delta

    # --- 4. Apply deltas to global structures ---
    for pair, delta in delta_local.items():
        global_delta = delta * freq
        new_global = pair_counts.get(pair, 0) + global_delta

        if new_global <= 0:
            pair_counts.pop(pair, None)
            if pair in pair_to_words:
                pair_to_words[pair].discard(word_id)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]
        else:
            pair_counts[pair] = new_global
            pair_to_words.setdefault(pair, set()).add(word_id)
            if heap is not None and vocab is not None:
                push_heap(heap, pair, new_global, vocab)

    # --- 5. Persist the updated token list and local counts on the Word ---
    word.tokens = new_tokens
    word.pair_counts = new_local


# ---------------------------------------------------------------------------
# GPT-2 pre-tokenization pattern
# ---------------------------------------------------------------------------

GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

#: Pre-compiled GPT-2 pattern for repeated use.
_COMPILED_GPT2 = re.compile(GPT2_PATTERN)

#: Maximum text length (in characters) for which ``regex.findall()`` is used
#: instead of the streaming ``regex.finditer()``.  ``findall`` is ~25 % faster
#: but materialises every match into a list.  The threshold is set conservatively
#: so the intermediate list stays under roughly 600 MB of resident memory.
_FINDALL_THRESHOLD_CHARS = 50_000_000  # 50 MB of UTF-8 text

#: Default number of worker processes for parallel pre-tokenization.
#: Capped at 8 to avoid diminishing returns from IPC overhead.
_DEFAULT_NUM_WORKERS = min(os.cpu_count() or 4, 8)

#: Minimum text length (characters) to trigger parallel pre-tokenization.
#: Below this threshold the process-spawn overhead dominates the regex savings.
_PARALLEL_THRESHOLD_CHARS = 10_000_000  # 10 MB


# ===================================================================
#  Corpus I/O
# ===================================================================

def load_text(path: str | Path) -> str:
    """Read a UTF-8 text file into a single string using memory-mapped I/O.

    Memory-mapping avoids an extra copy through Python's heap, but the
    returned string still occupies resident memory proportional to the
    file size.  For multi-gigabyte corpora this is the dominant fixed cost.

    Args:
        path: Path to the input text file (UTF-8 encoded).

    Returns:
        The entire file contents as a single string.
    """
    path = Path(path)
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.read().decode("utf-8")


# ===================================================================
#  Pre-tokenization
# ===================================================================

def pretokenize(
    text: str,
    special_tokens: list[str] | None = None,
) -> list[str]:
    """Split raw text into pre-tokens using the GPT-2 regex pattern.

    When *special_tokens* are provided they are kept intact as atomic units
    — they are never split further by the GPT-2 pattern, even if the pattern
    would otherwise match substrings inside them.

    .. note::
        This function materialises all pre-tokens into a list.  For training
        on large corpora, :func:`train_bpe` uses a streaming variant
        internally to avoid the memory cost of storing every occurrence.

    Args:
        text: Raw input text.
        special_tokens: Optional list of special-token strings to preserve
            as atomic units (e.g. ``["<|endoftext|>"]``).

    Returns:
        A flat list of pre-token strings.
    """
    if not special_tokens:
        return _COMPILED_GPT2.findall(text)

    # Build an alternation pattern that captures each special token as a
    # whole, then split on it.  Everything between special tokens is
    # re-tokenized with the standard GPT-2 pattern.
    escaped = "|".join(re.escape(tok) for tok in special_tokens)
    parts = re.split(f"({escaped})", text)
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            tokens.append(part)
        else:
            tokens.extend(_COMPILED_GPT2.findall(part))
    return tokens


# ===================================================================
#  Vocabulary initialisation
# ===================================================================

def initialize_vocab(
    special_tokens: list[str],
) -> dict[int, bytes]:
    """Create the initial BPE vocabulary.

    The vocabulary always starts with the 256 single-byte tokens (IDs 0--255),
    each mapping to the corresponding single-byte ``bytes`` object.  Special
    tokens are appended with IDs starting at 256.

    Args:
        special_tokens: List of special-token strings
            (e.g. ``["<|endoftext|>"]``).

    Returns:
        A dictionary ``{token_id: bytes}`` ready for BPE training.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab


# ===================================================================
#  Token sequence counting (adaptive strategy)
# ===================================================================

def _count_token_sequences(
    text: str,
    special_bytes: set[bytes],
) -> Counter[bytes]:
    """Count byte-encoded pre-token frequencies in *text*.

    Uses ``findall`` (faster, list-based) for texts under
    ``_FINDALL_THRESHOLD_CHARS`` and ``finditer`` (streaming) otherwise.

    Args:
        text: The raw corpus string.
        special_bytes: Set of special-token byte strings to exclude from
            the count (they are handled separately in the vocabulary).

    Returns:
        A :class:`~collections.Counter` mapping each ``bytes`` pre-token
        to its total frequency.
    """
    if len(text) < _FINDALL_THRESHOLD_CHARS:
        # Fast path: findall materialises a list but is ~25 % quicker.
        return Counter(
            token.encode("utf-8")
            for token in _COMPILED_GPT2.findall(text)
            if token.encode("utf-8") not in special_bytes
        )

    # Memory-safe path for large corpora: streaming generator.
    counter: Counter[bytes] = Counter()
    for m in _COMPILED_GPT2.finditer(text):
        seq = m.group().encode("utf-8")
        if seq not in special_bytes:
            counter[seq] += 1
    return counter


# ===================================================================
#  Parallel pre-tokenization (fork + copy-on-write)
# ===================================================================
#
# Sending large text chunks through :mod:`multiprocessing` pipes is slow
# (macOS pipe buffers are ~16 KB, so a 275 MB chunk needs ~17 000 round-
# trips).  Instead we store the pre-split chunk list in a module-level
# variable, fork the process, and pass only integer *indices* through the
# pool.  The forked children inherit the chunks via copy-on-write.

#: Set to ``True`` when ``fork`` start method is available on this platform.
_FORK_AVAILABLE = "fork" in mp.get_all_start_methods()

#: Pre-split text chunks, set before forking so children inherit via COW.
_fork_chunks: list[str] | None = None


def _worker_by_index(idx: int) -> Counter[bytes]:
    """Process the chunk at position *idx* in the module-level ``_fork_chunks``.

    Only the integer *idx* travels through the pool pipe — the chunk text
    is inherited from the parent via fork copy-on-write.

    Args:
        idx: Index into ``_fork_chunks``.

    Returns:
        A :class:`~collections.Counter` of ``{bytes: freq}`` for this chunk.
    """
    assert _fork_chunks is not None, "_fork_chunks must be set before forking"
    chunk_text = _fork_chunks[idx]
    compiled = re.compile(GPT2_PATTERN)

    if len(chunk_text) < _FINDALL_THRESHOLD_CHARS:
        return Counter(
            token.encode("utf-8")
            for token in compiled.findall(chunk_text)
        )

    counter: Counter[bytes] = Counter()
    for m in compiled.finditer(chunk_text):
        counter[m.group().encode("utf-8")] += 1
    return counter


def _split_text_into_chunks(text: str, num_chunks: int) -> list[str]:
    """Split *text* into roughly equal slices at newline (``\\n``) boundaries.

    Splitting at newlines is **safe** for GPT-2 pre-tokenization because
    none of the pattern's alternatives span a newline — ``\\n`` is always
    consumed by the ``\\s+`` alternative and treated as a standalone token.

    Args:
        text: The full corpus string.
        num_chunks: Target number of chunks.

    Returns:
        A list of text slices.  May contain fewer than *num_chunks* items
        if the text has insufficient newlines.
    """
    if num_chunks <= 1:
        return [text]

    total_len = len(text)
    chunk_size = total_len // num_chunks
    chunks: list[str] = []
    start = 0

    for i in range(num_chunks):
        if start >= total_len:
            break
        if i == num_chunks - 1:
            chunks.append(text[start:])
            break

        end = start + chunk_size
        # Walk forward to the nearest newline
        nl = text.find("\n", end)
        if nl == -1:
            chunks.append(text[start:])
            break
        end = nl + 1  # include the newline in this chunk
        chunks.append(text[start:end])
        start = end

    return [c for c in chunks if c]


def _count_token_sequences_parallel(
    text: str,
    num_workers: int,
) -> Counter[bytes]:
    """Count pre-token frequencies using multiple processes via ``fork``.

    The text is split into chunks and stored in the module-level
    ``_fork_chunks``.  After forking, workers access their chunk by index
    (an integer that fits in a single pipe buffer), avoiding the pipe
    bottleneck of sending multi-megabyte strings.

    Args:
        text: The raw corpus string (at least ``_PARALLEL_THRESHOLD_CHARS``).
        num_workers: Number of worker processes to spawn.

    Returns:
        A merged :class:`~collections.Counter` of ``{bytes: freq}``.
    """
    global _fork_chunks
    _fork_chunks = _split_text_into_chunks(text, num_workers)

    ctx = mp.get_context("fork")
    merged: Counter[bytes] = Counter()

    with ctx.Pool(processes=num_workers) as pool:
        # Each worker receives an integer index, not a string chunk
        indices = list(range(len(_fork_chunks)))
        for result in pool.imap_unordered(_worker_by_index, indices, chunksize=1):
            merged.update(result)

    return merged


# ===================================================================
#  Training entry-point
# ===================================================================

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    output_dir: str | Path | None = None,
    num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-pair encoding tokenizer on the corpus at *input_path*.

    This is the main entry point for BPE training.  It loads the corpus,
    pre-tokenizes it with the GPT-2 pattern, initializes the single-byte
    vocabulary, then iteratively merges the most frequent adjacent token
    pair until the target *vocab_size* is reached (or no more pairs exist).

    Args:
        input_path: Path to the training corpus (UTF-8 text file).
        vocab_size: Target vocabulary size (includes the 256 initial
            single-byte tokens and any *special_tokens*).
        special_tokens: Special token strings to include
            (e.g. ``["<|endoftext|>"]``).  They are never split during
            pre-tokenization and are added to the vocabulary
            *before* training starts.
        output_dir: If provided, :func:`save_tokenizer` is called after
            training to persist ``vocab.json`` and ``merges.txt`` into
            this directory.
        num_workers: Number of worker processes for parallel pre-tokenization.
            Pass ``None`` to auto-detect (``min(cpu_count, 8)``).  Parallel
            mode only activates when the text exceeds 10 MB.

    Returns:
        A tuple ``(vocab, merges)``:

        * **vocab** -- ``{token_id: bytes}`` vocabulary.
        * **merges** -- Ordered list of ``(token_a_bytes, token_b_bytes)``
          pairs in the order they were merged.
    """
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    # ---- 1. Load the corpus ----
    t0 = time.perf_counter()
    text = load_text(input_path)
    timings["load_text"] = time.perf_counter() - t0

    # ---- 2. Initialize vocabulary ----
    vocab = initialize_vocab(special_tokens)

    # ---- 3. Pre-tokenization → frequency counter ----
    # Uses an adaptive strategy with three tiers:
    #   (a) parallel multi-process for texts ≥ 10 MB (when num_workers > 1)
    #   (b) findall (fast, list-based) for texts < 50 MB
    #   (c) finditer (streaming) for large single-threaded texts
    t0 = time.perf_counter()
    special_bytes = {t.encode("utf-8") for t in special_tokens}

    # Resolve number of workers and decide serial vs parallel
    if num_workers is None:
        num_workers = _DEFAULT_NUM_WORKERS
    use_parallel = (
        _FORK_AVAILABLE
        and num_workers > 1
        and len(text) >= _PARALLEL_THRESHOLD_CHARS
    )

    if not special_tokens:
        if use_parallel:
            token_sequences = _count_token_sequences_parallel(text, num_workers)
        else:
            token_sequences = _count_token_sequences(text, special_bytes)
    else:
        # Only use the expensive regex-split path when special tokens
        # actually appear in the text.  A fast substring check avoids a
        # full regex scan when the tokens aren't present (the common case).
        needs_split = any(tok in text for tok in special_tokens)
        if not needs_split:
            if use_parallel:
                token_sequences = _count_token_sequences_parallel(text, num_workers)
            else:
                token_sequences = _count_token_sequences(text, special_bytes)
        else:
            # Special tokens present — split first (serial), then count
            # each part.  The split already breaks the text into smaller
            # pieces, so parallelism here would have diminishing returns.
            escaped = "|".join(re.escape(tok) for tok in special_tokens)
            parts = re.split(f"({escaped})", text)
            token_sequences = Counter()
            for part in parts:
                if not part or part in special_tokens:
                    continue
                token_sequences.update(
                    _count_token_sequences(part, special_bytes)
                )

    timings["pretokenize+count"] = time.perf_counter() - t0

    # ---- 4. Build incremental data structures + heap ----
    t0 = time.perf_counter()
    words_dict, pair_counts, pair_to_words = build_structures(dict(token_sequences))
    heap = init_heap(pair_counts, vocab)
    timings["build_structures"] = time.perf_counter() - t0

    # ---- 5. Main training loop ----
    t0 = time.perf_counter()
    merges: list[tuple[bytes, bytes]] = []

    initial_vocab_size = len(vocab)
    pbar = tqdm(
        total=vocab_size,
        initial=initial_vocab_size,
        desc="Training BPE",
        unit="token",
        dynamic_ncols=True,
    )

    merge_count = 0
    while len(vocab) < vocab_size:
        # Pop the highest-priority pair (stale entries are auto-skipped)
        best_pair = pop_best_pair(heap, pair_counts, vocab)
        if best_pair is None:
            pbar.write("No more pairs to merge — stopping early.")
            break

        a, b = best_pair
        new_id = len(vocab)

        # Record and apply the merge
        merges.append((vocab[a], vocab[b]))
        vocab[new_id] = vocab[a] + vocab[b]
        merge_count += 1
        pbar.update(1)

        # Propagate the merge through every affected word
        affected = list(pair_to_words.get(best_pair, []))
        for wid in affected:
            word = words_dict.get(wid)
            if word is None:
                continue
            update_word(
                word, a, b, new_id,
                pair_counts, pair_to_words, wid,
                heap, vocab,
            )

        # The merged pair no longer exists — clean up the inverted index
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]

    pbar.close()
    timings["training_loop"] = time.perf_counter() - t0

    # ---- 6. Optionally persist to disk ----
    if output_dir is not None:
        t0 = time.perf_counter()
        save_tokenizer(vocab, merges, output_dir)
        timings["save"] = time.perf_counter() - t0
        pbar.write(f"Tokenizer saved to {Path(output_dir).resolve()}")

    # ---- Report timings ----
    timings["total"] = time.perf_counter() - t_start
    _print_timings(timings, merge_count, num_workers if use_parallel else 1)

    return vocab, merges


def _print_timings(timings: dict[str, float], merge_count: int, num_workers: int = 1) -> None:
    """Print a formatted timing summary after training."""
    total = timings.get("total", 1.0)
    print("\n" + "=" * 52)
    print(f"{'Phase':<25} {'Time (s)':>10} {'%':>8}")
    print("-" * 52)
    for label in ("load_text", "pretokenize+count", "build_structures", "training_loop", "save"):
        if label in timings:
            t = timings[label]
            pct = 100.0 * t / total
            print(f"{label:<25} {t:>10.2f} {pct:>7.1f}%")
    print("-" * 52)
    print(f"{'TOTAL':<25} {total:>10.2f}")
    if merge_count:
        print(f"{'Merges performed':<25} {merge_count:>10}")
    if num_workers > 1:
        print(f"{'Workers (parallel)':<25} {num_workers:>10}")
    print("=" * 52)


import base64
import json


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
    vocab_filepath: str,
    merges_filepath: str
) -> None:
    os.makedirs(os.path.dirname(vocab_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(merges_filepath), exist_ok=True)
    
    # 将 vocab 的键转为字符串，值转为 Base64 字符串
    vocab_b64 = {str(k): base64.b64encode(v).decode('ascii') for k, v in vocab.items()}
    # 将 merges 的每个 bytes 对转为 Base64 字符串对
    merges_b64 = [
        (base64.b64encode(left).decode('ascii'), base64.b64encode(right).decode('ascii'))
        for left, right in merges
    ]

    vocab_data = {
        "special_tokens": special_tokens,
        "vocab": vocab_b64
    }
    merges_data = {"merges": merges_b64}

    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)   # indent 可选，便于阅读
    with open(merges_filepath, "w", encoding="utf-8") as f:
        json.dump(merges_data, f, indent=2)


def load_tokenizer(
    vocab_filepath: str,
    merges_filepath: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], list[str]]:
    """
    从两个 JSON 文件加载 tokenizer 数据，返回 (vocab, merges, special_tokens)。
    """
    with open(vocab_filepath, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    with open(merges_filepath, "r", encoding="utf-8") as f:
        merges_data = json.load(f)

    # 还原 vocab：键从字符串转回 int，值从 Base64 解码为 bytes
    vocab = {
        int(k): base64.b64decode(v.encode('ascii'))
        for k, v in vocab_data["vocab"].items()
    }
    # 还原 merges：每个 Base64 字符串对解码为 bytes 对
    merges = [
        (base64.b64decode(left.encode('ascii')), base64.b64decode(right.encode('ascii')))
        for left, right in merges_data["merges"]
    ]
    special_tokens = vocab_data.get("special_tokens", []) 

    return vocab, merges, special_tokens