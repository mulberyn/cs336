"""BPE tokenizer training pipeline.

This module bundles everything needed to train a byte-pair encoding tokenizer
from a raw text corpus and persist the result to disk:

* Corpus loading via memory-mapped I/O.
* GPT-2-style pre-tokenization (with special-token preservation).
* Vocabulary initialisation (single-byte tokens + specials).
* The main incremental training loop with heap-based pair selection.
* Serialization helpers that write the final vocabulary and merge table
  using hex-encoded byte strings for unambiguous round-tripping.

Performance note
----------------
The pipeline uses two key optimisations for large corpora:

1. **Streaming pre-tokenization** — ``regex.finditer()`` is used as a
   generator so that pre-token strings never materialise as a giant list.
   Frequencies are counted directly into a :class:`collections.Counter`
   keyed by ``bytes``.
2. **Bytes dictionary keys** — token sequences are stored as ``bytes``
   objects (hashable, compact) rather than ``tuple[int, ...]``, avoiding
   an expensive per-word conversion and reducing memory pressure.
"""

import json
import mmap
import time
from collections import Counter
from pathlib import Path

import regex as re
from tqdm import tqdm

from .bpe import build_structures, init_heap, pop_best_pair, update_word

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
#  Serialization
# ===================================================================

def _bytes_to_repr(b: bytes) -> str:
    """Encode an arbitrary byte string as a compact hex string.

    Each byte becomes two lowercase hex characters (e.g. ``b"ab"`` → ``"6162"``).
    This representation is unambiguous and survives any text encoding.

    Args:
        b: The byte string to encode.

    Returns:
        Hex string representation.
    """
    return b.hex()


def _repr_to_bytes(s: str) -> bytes:
    """Decode a hex string back into the original byte string.

    Args:
        s: Hex string produced by :func:`_bytes_to_repr`.

    Returns:
        The original byte string.
    """
    return bytes.fromhex(s)


def save_vocab(
    vocab: dict[int, bytes],
    path: str | Path,
) -> None:
    """Save the tokenizer vocabulary to a JSON file.

    Each entry is serialized as ``{token_id: hex_string}``.  Token IDs are
    written as decimal strings (JSON requires string keys), and token bytes
    are written as hex strings.

    Args:
        vocab: ``{token_id: bytes}`` vocabulary mapping.
        path: Output file path (typically ``vocab.json``).
    """
    serialized: dict[str, str] = {}
    for tid, token_bytes in sorted(vocab.items()):
        serialized[str(tid)] = _bytes_to_repr(token_bytes)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)
    # Trailing newline — good citizenship for text files
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n")


def save_merges(
    merges: list[tuple[bytes, bytes]],
    path: str | Path,
) -> None:
    """Save the ordered merge table to a text file.

    Each line contains two space-separated hex-encoded token strings,
    representing one merge ``(token_a, token_b)`` in the order they were
    created during training.  The first line is the first merge.

    Args:
        merges: Ordered list of ``(token_a_bytes, token_b_bytes)`` pairs.
        path: Output file path (typically ``merges.txt``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_bytes_to_repr(a)} {_bytes_to_repr(b)}\n")


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | Path,
) -> None:
    """Persist a trained BPE tokenizer to disk.

    Creates two files under *output_dir*:

    * ``vocab.json`` — hex-encoded vocabulary (JSON).
    * ``merges.txt`` — hex-encoded merge rules (plain text, one per line).

    The directory is created if it does not exist.

    Args:
        vocab: ``{token_id: bytes}`` vocabulary.
        merges: Ordered list of ``(token_a, token_b)`` merge pairs.
        output_dir: Directory in which to write the two files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, output_dir / "vocab.json")
    save_merges(merges, output_dir / "merges.txt")


# ===================================================================
#  Training entry-point
# ===================================================================

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    output_dir: str | Path | None = None,
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
    # Uses an adaptive strategy: findall (fast) for texts under 50 MB,
    # finditer (streaming) for larger corpora.  Both paths produce a
    # Counter keyed by bytes — no intermediate list of token strings
    # is kept alive after this phase.
    t0 = time.perf_counter()
    special_bytes = {t.encode("utf-8") for t in special_tokens}

    if not special_tokens:
        token_sequences = _count_token_sequences(text, special_bytes)
    else:
        # Only use the expensive regex-split path when special tokens
        # actually appear in the text.  A fast substring check avoids a
        # full regex scan when the tokens aren't present (the common case).
        needs_split = any(tok in text for tok in special_tokens)
        if not needs_split:
            token_sequences = _count_token_sequences(text, special_bytes)
        else:
            escaped = "|".join(re.escape(tok) for tok in special_tokens)
            parts = re.split(f"({escaped})", text)
            token_sequences = Counter()
            for part in parts:
                if not part or part in special_tokens:
                    continue
                token_sequences.update(_count_token_sequences(part, special_bytes))

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
    _print_timings(timings, merge_count)

    return vocab, merges


def _print_timings(timings: dict[str, float], merge_count: int) -> None:
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
    print("=" * 52)


# ===================================================================
#  CLI demo  (python -m cs336_basics.tokenizer.train)
# ===================================================================
if __name__ == "__main__":
    INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
    VOCAB_SIZE = 10000
    SPECIAL_TOKENS = ["<|endoftext|>"]
    OUTPUT_DIR = "out/tokenizer"

    print(f"Training BPE tokenizer on {INPUT_PATH} …")
    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        output_dir=OUTPUT_DIR,
    )
    print(f"\n  Vocabulary size: {len(vocab)}")
    print(f"  Merge rules:     {len(merges)}")
    print(f"  Output:          {OUTPUT_DIR}/vocab.json, {OUTPUT_DIR}/merges.txt")
