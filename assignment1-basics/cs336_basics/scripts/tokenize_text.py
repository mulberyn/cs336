import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

from cs336_basics.tokenizer import Tokenizer

DEFAULT_FILE_DIR = "./data"
DEFAULT_TOEKENIZER_DIR = "./out/tokenizer"


def load_prase():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="待 tokenize 文件名")
    parser.add_argument("--tokenizer-dir", type=str, help="tokenizer 所在目录")
    args = parser.parse_args()
    return args


def load_tokenizer(
    tokenizer_dir: str
) -> Tokenizer:
    return Tokenizer.from_file(
        vocab_filepath=Path(tokenizer_dir) / "vocab.json",
        merges_filepath=Path(tokenizer_dir) / "merges.txt"
    )


def main():
    args = load_prase()
    tokenizer = load_tokenizer(DEFAULT_TOEKENIZER_DIR)
    filename = args.filename
    
    output_path = DEFAULT_FILE_DIR / filename / '.bin'
    with open(output_path, 'wb') as f_out:
        with open(DEFAULT_FILE_DIR / filename, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, desc="Tokenizing", unit="lines"):
                if not line.strip():
                    continue
                ids = tokenizer.encode(line.rstrip('\n'))
                ids_array = np.array(ids, dtype=np.uint32)
                f_out.write(ids_array.tobytes())


if __name__ == "__main__":
    sys.exit(main())