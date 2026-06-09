from .utils import load_text
from .pretokenize import pretokenize
from .vocabulary import initialize_vocab
from .bpe import build_structures, update_word, init_heap, pop_best_pair


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 加载并预分词
    text = load_text(input_path)
    words = pretokenize(text, special_tokens)

    # 初始化词汇表
    vocab = initialize_vocab(special_tokens)

    # 构建 token_sequences（整数元组 -> 频率）
    token_sequences = {}
    for w in words:
        if w in special_tokens:
            continue
        seq = tuple(w.encode('utf-8'))
        token_sequences[seq] = token_sequences.get(seq, 0) + 1

    # 构建增量数据结构
    words_dict, pair_counts, pair_to_words = build_structures(token_sequences)

    # Stage 4: 初始化堆
    heap = init_heap(pair_counts, vocab)

    merges = []

    # 训练循环
    while len(vocab) < vocab_size:
        # 从堆中获取当前最佳 pair
        best_pair = pop_best_pair(heap, pair_counts, vocab)
        if best_pair is None:
            break   # 没有可合并的 pair

        a, b = best_pair
        new_id = len(vocab)

        # 记录合并
        merges.append((vocab[a], vocab[b]))
        vocab[new_id] = vocab[a] + vocab[b]

        # 获取所有包含该 pair 的单词 ID（拷贝，避免遍历时修改）
        affected = list(pair_to_words.get(best_pair, []))
        for wid in affected:
            word = words_dict.get(wid)
            if word is None:
                continue
            # 更新单词，传入 heap 和 vocab 以便内部 push 新计数
            update_word(word, a, b, new_id, pair_counts, pair_to_words, wid, heap, vocab)

        # 合并后该 pair 应不再存在，删除倒排索引条目（pair_counts 已由 update_word 置 0 并删除）
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]

    return vocab, merges


if __name__ == "__main__":
    from tests.adapters import run_train_bpe

    input_path = "data/sample.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=257 + 12,
        special_tokens=["<|endoftext|>"],
    )
    print("Learned vocab:")
    for token_id, token_bytes in vocab.items():
        print(f"{token_id}: {token_bytes}")
    print("\nLearned merges:")
    for merge in merges:  
        print(merge)
    