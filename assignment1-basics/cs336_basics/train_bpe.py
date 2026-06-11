#!/usr/bin/env python3
"""
Train a BPE tokenizer on one or more text files.

This script uses the BPE training implementation from cs336_basics.tokenizer
and saves the resulting vocabulary and merge operations to disk.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 假设 cs336_basics 包在 PYTHONPATH 中，或者按相对路径导入
from cs336_basics.tokenizer import train_bpe

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def save_vocab(vocab: Dict[int, bytes], output_path: Path) -> None:
    """
    Save vocabulary to a JSON file.

    Args:
        vocab: Mapping from token id to byte representation.
        output_path: File path to save the vocab.
    """
    # 将 bytes 转换为十六进制字符串以便 JSON 序列化
    serializable = {str(idx): val.hex() for idx, val in vocab.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved vocabulary to {output_path}")


def save_merges(merges: List[Tuple[bytes, bytes]], output_path: Path) -> None:
    """
    Save merge operations to a text file.

    Args:
        merges: List of pairs of byte sequences that were merged.
        output_path: File path to save the merges (one pair per line).
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            # 将 bytes 转换为十六进制或使用 repr 来安全存储
            # 这里使用 repr（不推荐用于生产，但简单），也可用 hex
            f.write(f"{a.hex()} {b.hex()}\n")
    logger.info(f"Saved merges to {output_path}")


def train_on_file(
    input_path: Path,
    vocab_size: int,
    special_tokens: List[str],
    output_dir: Path,
) -> None:
    """
    Train a BPE tokenizer on a single file and save results.

    Args:
        input_path: Path to the training text file.
        vocab_size: Desired vocabulary size (includes special tokens).
        special_tokens: List of special token strings.
        output_dir: Directory where vocab.json and merges.txt will be written.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training on {input_path} (vocab_size={vocab_size})")
    try:
        vocab, merges = train_bpe(
            input_path=str(input_path),
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )
    except Exception as e:
        logger.error(f"Training failed for {input_path}: {e}")
        raise

    # 保存结果
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    logger.info(
        f"Finished training on {input_path}. Final vocab size: {len(vocab)}, "
        f"number of merges: {len(merges)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer on one or more text files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Path(s) to training text file(s).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Target vocabulary size (including special tokens). Default: 10000.",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=["<|endoftext|>"],
        help="List of special tokens to include. Default: <|endoftext|>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./bpe_model"),
        help="Base output directory. For each input file, a subdirectory "
             "named after the file stem will be created. Default: ./bpe_model.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level.",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    # 对每个输入文件分别训练
    for input_path in args.inputs:
        # 为每个文件创建独立的输出子目录，例如 owt_train -> ./bpe_model/owt_train/
        subdir = args.output_dir / input_path.stem
        train_on_file(
            input_path=input_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            output_dir=subdir,
        )


if __name__ == "__main__":
    main()