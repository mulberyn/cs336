from cs336_basics.tokenizer.ctypes import Word
from collections import Counter, defaultdict


def build_from_token_sequences(
    token_sequences: dict[tuple[int], int]
) -> tuple[dict[int, Word], dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
    """
    input: 
        token_sequences: dict { (t1, t2, ...): freq }
        vocab: dict { token_id: token_bytes }
    output:
        words: dict { word_id: Word(tokens, freq) } 每个 word_id 对应一个 Word 对象，包含 token 列表和频率
        pair_counts: dict { (a, b): total_freq } 'a' 和 'b' 组成的 pair 在所有单词中出现的总频率
        pair_to_words: dict { (a, b): set(word_id) } 'a' 和 'b' 组成的 pair 出现在哪些单词中
    """
    
    words = {}
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    word_id = 0
    
    for seq, freq in token_sequences.items():
        tokens = list(seq)
        word = Word(tokens, freq)
        words[word_id] = word
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_id)
        word_id += 1
    
    return words, pair_counts, pair_to_words
    

def update_word(
    word: Word,
    a: int, b: int,
    new_id: int,
    pair_counts: dict[tuple[int, int], int],
    pair_to_words: dict[tuple[int, int], set[int]],
    word_id: int,
):
    """
    word: Word 对象，该 word 中包含需要被合并的 pair (a, b)
    a, b: 需要被合并的 token_id(int)
    new_id: 新 token 的 token_id(int)
    pair_counts: 全局 pair 频率统计
    pair_to_words: 全局 pair 到 word_id 的倒排索引
    word_id: 当前 word 的 id(int)，用于更新倒排索引
    """
    
    old_tokens = word.tokens
    freq = word.freq

    # 更新 word.tokens 列表
    new_tokens = []
    i = 0
    while i < len(old_tokens):
        if i < len(old_tokens) - 1 and old_tokens[i] == a and old_tokens[i + 1] == b:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(old_tokens[i])
            i += 1

    # 用 new_local 更新 word.pair_counts
    new_local = Counter()
    for i in range(len(new_tokens) - 1):
        new_local[(new_tokens[i], new_tokens[i + 1])] += 1

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