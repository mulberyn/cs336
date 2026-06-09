import heapq
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

# ----------------------------------------------------------------------
# 辅助函数（与原始实现保持一致）
# ----------------------------------------------------------------------
def load_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def pretokenize(text: str, special_tokens: List[str]) -> List[str]:
    """简单的预分词：按空格分割，但保留特殊 token 作为整体"""
    words = []
    for word in text.split():
        if word in special_tokens:
            words.append(word)
        else:
            # 这里可根据需要加入更复杂的分词逻辑（如 GPT-2 的正则）
            # 为简单起见，直接返回单词本身
            words.append(word)
    return words

def initialize_vocab(special_tokens: List[str]) -> Dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode('utf-8')
    return vocab

# ----------------------------------------------------------------------
# Stage 3 所需的数据结构
# ----------------------------------------------------------------------
class Word:
    __slots__ = ('tokens', 'freq', 'pair_counts')
    def __init__(self, tokens: List[int], freq: int):
        self.tokens = tokens          # list of int token ids
        self.freq = freq              # global frequency of this word
        # 局部 pair 计数（该单词内部每个相邻 pair 的出现次数）
        self.pair_counts = Counter()
        for i in range(len(tokens) - 1):
            self.pair_counts[(tokens[i], tokens[i+1])] += 1

# ----------------------------------------------------------------------
# 从 token_sequences 构建增量数据结构
# ----------------------------------------------------------------------
def build_structures(
    token_sequences: Dict[Tuple[int, ...], int]
) -> Tuple[Dict[int, Word], Dict[Tuple[int, int], int], Dict[Tuple[int, int], Set[int]]]:
    """
    token_sequences: { (t1, t2, ...): freq }
    返回: words, pair_counts, pair_to_words
    """
    words = {}
    pair_counts = Counter()
    pair_to_words = defaultdict(set)

    word_id = 0
    for seq, freq in token_sequences.items():
        tokens = list(seq)
        word = Word(tokens, freq)
        words[word_id] = word
        # 更新全局统计
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_id)
        word_id += 1
    return words, pair_counts, pair_to_words

# ----------------------------------------------------------------------
# 寻找最佳 pair（完全复现原始比较逻辑）
# ----------------------------------------------------------------------
def find_best_pair(
    pair_counts: Dict[Tuple[int, int], int],
    vocab: Dict[int, bytes]
) -> Tuple[int, int]:
    if not pair_counts:
        raise ValueError("No pairs to choose from")
    # 排序键: (频率, (vocab[a], vocab[b]))
    # 频率高的优先；频率相同时，比较 token 的 bytes 元组（字典序，取较大者）
    best_pair = max(
        pair_counts.items(),
        key=lambda item: (item[1], (vocab[item[0][0]], vocab[item[0][1]]))
    )[0]
    return best_pair

# ----------------------------------------------------------------------
# 增量更新一个单词（核心）
# ----------------------------------------------------------------------
def update_word(
    word: Word,
    a: int, b: int,
    new_id: int,
    pair_counts: Dict[Tuple[int, int], int],
    pair_to_words: Dict[Tuple[int, int], Set[int]],
    word_id: int
) -> None:
    """
    将单词中所有出现的 (a, b) 替换为 new_id，并同步更新全局 pair_counts 和 pair_to_words。
    """
    old_tokens = word.tokens
    freq = word.freq

    # 1. 构建新的 token 列表
    new_tokens = []
    i = 0
    while i < len(old_tokens):
        if i < len(old_tokens) - 1 and old_tokens[i] == a and old_tokens[i+1] == b:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(old_tokens[i])
            i += 1

    # 2. 统计新 tokens 的局部 pair 计数
    new_local = Counter()
    for i in range(len(new_tokens) - 1):
        new_local[(new_tokens[i], new_tokens[i+1])] += 1

    # 3. 旧局部 pair 计数（直接从 word.pair_counts 获取）
    old_local = word.pair_counts

    # 4. 计算每个 pair 的局部变化量
    all_pairs = set(old_local.keys()) | set(new_local.keys())
    delta_local = {}
    for pair in all_pairs:
        delta = new_local.get(pair, 0) - old_local.get(pair, 0)
        if delta != 0:
            delta_local[pair] = delta

    # 5. 应用全局更新
    for pair, delta in delta_local.items():
        global_delta = delta * freq
        old_global = pair_counts.get(pair, 0)
        new_global = old_global + global_delta

        if new_global <= 0:
            # 删除该 pair 的全局记录
            if pair in pair_counts:
                del pair_counts[pair]
            # 从倒排索引中移除当前单词
            if pair in pair_to_words:
                pair_to_words[pair].discard(word_id)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]
        else:
            pair_counts[pair] = new_global
            # 确保倒排索引包含当前单词
            pair_to_words.setdefault(pair, set()).add(word_id)

    # 6. 更新单词对象
    word.tokens = new_tokens
    word.pair_counts = new_local

# ----------------------------------------------------------------------
# 增量训练主函数
# ----------------------------------------------------------------------
def train_bpe_incremental(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 加载并预分词
    text = load_text(input_path)
    words_list = pretokenize(text, special_tokens)

    # 初始化词汇表（0~255 + 特殊 token）
    vocab = initialize_vocab(special_tokens)

    # 构建 token_sequences（整数元组 -> 频率）
    token_sequences = {}
    for w in words_list:
        print(w)
        if w in special_tokens:
            continue
        seq = tuple(w.encode('utf-8'))   # 每个字节变成 0~255 的整数
        token_sequences[seq] = token_sequences.get(seq, 0) + 1

    # 构建增量数据结构
    words, pair_counts, pair_to_words = build_structures(token_sequences)

    merges = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 选择最佳 pair
        best_pair = find_best_pair(pair_counts, vocab)
        a, b = best_pair
        new_id = len(vocab)

        # 记录合并（bytes 形式）
        merges.append((vocab[a], vocab[b]))
        vocab[new_id] = vocab[a] + vocab[b]

        # 获取所有包含该 pair 的单词 id（拷贝，因为遍历时会修改字典）
        affected = list(pair_to_words.get(best_pair, []))
        for wid in affected:
            word = words[wid]
            # 单词可能已经被之前的合并删除（理论上不会，但防御）
            if word is None:
                continue
            update_word(word, a, b, new_id, pair_counts, pair_to_words, wid)

        # 合并完成后，该 pair 不再存在于任何单词中，删除倒排索引条目
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]
        # pair_counts 中的条目已被 update_word 处理并删除（因为 new_global <= 0）

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe_incremental(
        input_path = "data/sample.txt",
        vocab_size = 257 + 12,
        special_tokens = ["<|endoftext|>"],
    )
    print("Vocabulary:")
    for idx, token in vocab.items():
        print(f"{idx}: {token}")
    print("\nMerges:")
    for a, b in merges:
        print(f"{a} + {b} -> {a + b}")