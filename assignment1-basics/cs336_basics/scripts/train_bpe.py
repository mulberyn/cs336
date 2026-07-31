import argparse
import sys
from pathlib import Path

from cs336_basics.tokenizer import train_bpe, save_tokenizer

DEFAULT_INPUT_PATH = "./data/TinyStoriesV2-GPT4-train.txt"
DEFAULT_OUT_DIR = "./out/tokenizer"


def load_prase():
    parser = argparse.ArgumentParser(
        description="在原始文本语料库上训练 BPE tokenizer。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python train_bpe.py data/train.txt --vocab-size 10000 --special-tokens <|endoftext|>"
    )
    parser.add_argument("--input-path", type=str, default=DEFAULT_INPUT_PATH, help="UTF-8 训练语料库路径")
    parser.add_argument("--vocab-size", type=int, default=10000, help="目标词表大小（≥ 256）")
    parser.add_argument("--special-tokens", nargs="*", default=['<|endoftext|>'], help="特殊 token 列表，例如 <|endoftext|>")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR, help="输出目录（默认 ./out/tokenizer）")
    parser.add_argument("--num-workers", type=int, default=1, help="并行预分词 worker 数（默认 1）")
    args = parser.parse_args()
    return args


def validate_args(args):
    input_path = Path(args.input_path)
    if not input_path.is_file():
        return False, f"输入文件 '{input_path}' 不存在"
    if args.vocab_size < 256 + len(args.special_tokens):
        return False, f"vocab_size 必须 >= {256 + len(args.special_tokens)}"
    return True, None


def main():
    args = load_prase()
    
    ok, msg = validate_args(args)
    if not ok:
        print(f'错误{msg}', file=sys.stderr)
        return 1
    
    try:
        vocab, merges = train_bpe(
            input_path=args.input_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
        )
    except Exception as e:
        print(f"训练失败：{e}", file=sys.stderr)
        return 1
    
    try:
        save_tokenizer(vocab, merges, args.special_tokens, vocab_filepath=args.output_dir + "/vocab.json", merges_filepath=args.output_dir + "/merges.json")
    except OSError as e:
        print(f"保存失败：{e}", file=sys.stderr)
        return 1
    
    print(f"训练完成，词表大小：{len(vocab)}，合并数：{len(merges)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())