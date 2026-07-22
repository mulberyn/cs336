"""Incremental BPE algorithm internals.

This module provides the core data structures and operations for training a
byte-pair encoding (BPE) tokenizer: building the initial word-indexed
structures, maintaining a max-heap of candidate merge pairs, and efficiently
updating word tokens and global pair counts after each merge.
"""

import heapq
from collections import Counter, defaultdict
from typing import Optional

from .core import Word


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


# ---------------------------------------------------------------------------
# Heap helpers
# ---------------------------------------------------------------------------

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
