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
    print(f"Loaded {len(words)} words from {input_path}.")

    # 初始化词汇表
    vocab = initialize_vocab(special_tokens)

    # 构建 token_sequences（整数元组 -> 频率）
    token_sequences = {}
    for w in words:
        if w in special_tokens:
            continue
        seq = tuple(w.encode('utf-8'))
        token_sequences[seq] = token_sequences.get(seq, 0) + 1

    print(f"Initial vocabulary size (excluding special tokens): {len(token_sequences)}")
    # 构建增量数据结构
    words_dict, pair_counts, pair_to_words = build_structures(token_sequences)

    # Stage 4: 初始化堆
    heap = init_heap(pair_counts, vocab)

    merges = []

    print(f"Starting BPE training to reach vocab size {vocab_size}...")
    
    # 训练循环
    while len(vocab) < vocab_size:
        print(f"Current vocab size: {len(vocab)}. Merges so far: {len(merges)}. Heap size: {len(heap)}.")
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
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    
    import time
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path = input_path,
        vocab_size = 10000,
        special_tokens = ["<|endoftext|>"],
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")