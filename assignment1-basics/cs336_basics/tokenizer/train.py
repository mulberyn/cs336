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
import os
import time
from collections import Counter
from pathlib import Path

import regex as re
from tqdm import tqdm

from .bpe import build_structures, init_heap, pop_best_pair, update_word

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def load_text(
    path: str | Path
) -> str:
    path = Path(path)
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.read().decode("utf-8")


def pretokenize(
    text: str,
    special_tokens: list[str] | None = None,
) -> list[str]:
    if not special_tokens:
        return re.findall(GPT2_PATTERN, text)

    escaped = "|".join(re.escape(tok) for tok in special_tokens)
    parts = re.split(f"({escaped})", text)
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            tokens.append(part)
        else:
            tokens.extend(re.findall(GPT2_PATTERN, part))
    return tokens


def initialize_vocab(
    special_tokens: list[str],
) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab


# ===================================================================
#  CLI demo  (uv run python -m cs336_basics.tokenizer.train)
# ===================================================================
if __name__ == "__main__":
    INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
    VOCAB_SIZE = 1000
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
