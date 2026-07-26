"""Core data structures for the BPE tokenizer."""

from collections import Counter


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
