import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

from cs336_basics.tokenizer import Tokenizer

DEFAULT_CONFIG = {
    'input_path': "./data/TinyStoriesV2-GPT4-train.txt",
    'output_path': './data/TinyStoriesV2-GPT4-train.bin',
    'tokenizer_dir': "./out/tokenizer",
}


def load_prase():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=DEFAULT_CONFIG["input_path"], help="待 tokenize 文件输入路径")
    parser.add_argument("--output_path", type=str, default=DEFAULT_CONFIG["output_path"], help="tokenize 文件输出路径")
    parser.add_argument("--tokenizer_dir", type=str, default=DEFAULT_CONFIG["tokenizer_dir"], help="tokenizer 所在目录")
    args = parser.parse_args()
    return args


def load_tokenizer(
    tokenizer_dir: str
) -> Tokenizer:
    return Tokenizer.from_file(
        vocab_filepath=Path(tokenizer_dir) / "vocab.json",
        merges_filepath=Path(tokenizer_dir) / "merges.json"
    )


def main():
    args = load_prase()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    
    input_path = args.input_path
    output_path = args.output_path
    
    with open(output_path, 'wb') as f_out:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, desc="Tokenizing", unit="lines"):
                if not line.strip():
                    continue
                ids = tokenizer.encode(line.rstrip('\n'))
                ids_array = np.array(ids, dtype=np.uint32)
                f_out.write(ids_array.tobytes())
    
    print(f"文件 {input_path} tokenize 完成，输出 {output_path}。")


if __name__ == "__main__":
    sys.exit(main())