"""CS336 BPE tokenizer package.

Provides a complete BPE tokenizer training pipeline:

* Corpus loading and GPT-2-style pre-tokenization.
* Incremental BPE training with a heap-based priority queue.
* Serialization of trained vocabularies and merge tables to disk.

Example usage::

    from cs336_basics.tokenizer import train_bpe

    vocab, merges = train_bpe(
        input_path="data/corpus.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        output_dir="out/tokenizer",
    )
"""

from .core import Word
from .train import (
    initialize_vocab,
    pretokenize,
    save_merges,
    save_tokenizer,
    save_vocab,
    train_bpe,
)

__all__ = [
    "train_bpe",
    "initialize_vocab",
    "pretokenize",
    "save_vocab",
    "save_merges",
    "save_tokenizer",
    "Word",
]
