from collections import Counter, defaultdict
from typing import Optional

from .core import Word
import heapq


def build_structures(
    token_sequences: dict[tuple[int, ...], int]
) -> tuple[dict[int, Word], dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
    """
    从 token_sequences 构建增量训练所需的数据结构。
    
    参数:
        token_sequences: { 整数元组: 频率 }
    返回:
        words:        { word_id: Word }
        pair_counts:  { (a,b): 全局总频次 }
        pair_to_words:{ (a,b): set of word_id }
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
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_id)
        word_id += 1

    return words, pair_counts, pair_to_words


class ReverseBytes:
    """包装字节串，使其在 Python 的小根堆中表现为逆序（字典序大的在前）"""
    __slots__ = ('data',)  # 使用 __slots__ 优化内存和加速属性访问

    def __init__(self, data: tuple[bytes, bytes]):
        self.data = data
        
    def __lt__(self, other: 'ReverseBytes') -> bool:
        # 颠倒小于号逻辑：自己实际大于别人时，返回 True（骗过小根堆，让大的排在前面）
        return self.data > other.data


def _make_heap_item(
    count: int, 
    pair: tuple[int, int], 
    vocab: dict[int, bytes]
) -> tuple:
    a, b = pair
    # 1. -count: 使得 count 大的在前
    # 2. ReverseBytes(vocab[a]): 使得 vocab[a] 字典序大的在前
    # 3. ReverseBytes(vocab[b]): 使得 vocab[b] 字典序大的在前
    # 4. a, b: 作为最后的保底标识，并用于解包
    return (-count, ReverseBytes((vocab[a], vocab[b])), a, b)


def init_heap(
    pair_counts: dict[tuple[int, int], int], 
    vocab: dict[int, bytes]
) -> list:
    """从初始 pair_counts 构建堆"""
    heap = [_make_heap_item(count, pair, vocab) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    return heap


def push_heap(
    heap: list, 
    pair: tuple[int, int], 
    new_count: int, 
    vocab: dict[int, bytes]
) -> None:
    """当 pair 的计数更新时，将新条目推入堆"""
    heapq.heappush(heap, _make_heap_item(new_count, pair, vocab))


def pop_best_pair(
    heap: list, 
    pair_counts: dict[tuple[int, int], int], 
    vocab: dict[int, bytes]
) -> Optional[tuple[int, int]]:
    
    while heap:
        # 这里的 _reversed_key 对应原来的五个元素，现在缩减回四个（neg_count, 包装类, a, b）
        neg_count, _reversed_key, a, b = heapq.heappop(heap)
        pair = (a, b)
        current = pair_counts.get(pair, 0)
        
        if current == -neg_count: 
            return pair
            
    return None


def update_word(
    word: Word,
    a: int, b: int,
    new_id: int,
    pair_counts: dict[tuple[int, int], int],
    pair_to_words: dict[tuple[int, int], set[int]],
    word_id: int,
    heap: Optional[list] = None,          # Stage 4: 堆对象，若提供则 push 新计数
    vocab: Optional[dict[int, bytes]] = None,  # Stage 4: 需要 vocab 构造堆条目
) -> None:
    """
    将单词中所有 (a,b) 替换为 new_id，并同步全局统计。
    若提供了 heap 和 vocab，则对每次 count 变化都 push 新堆条目。
    """
    old_tokens = word.tokens
    freq = word.freq

    # 1. 构建新 token 列表
    new_tokens = []
    i = 0
    while i < len(old_tokens):
        if i < len(old_tokens) - 1 and old_tokens[i] == a and old_tokens[i+1] == b:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(old_tokens[i])
            i += 1

    # 2. 新局部计数
    new_local = Counter()
    for i in range(len(new_tokens) - 1):
        new_local[(new_tokens[i], new_tokens[i+1])] += 1

    old_local = word.pair_counts

    # 3. 计算变化量
    all_pairs = set(old_local.keys()) | set(new_local.keys())
    delta_local = {}
    for pair in all_pairs:
        delta = new_local.get(pair, 0) - old_local.get(pair, 0)
        if delta != 0:
            delta_local[pair] = delta

    # 4. 应用全局更新，同时更新堆
    for pair, delta in delta_local.items():
        global_delta = delta * freq
        old_global = pair_counts.get(pair, 0)
        new_global = old_global + global_delta

        if new_global <= 0:
            # 删除全局记录
            if pair in pair_counts:
                del pair_counts[pair]
            if pair in pair_to_words:
                pair_to_words[pair].discard(word_id)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]
            # 不需要推入堆，因为计数为 0 会被忽略（堆中旧条目在 pop 时被跳过）
        else:
            pair_counts[pair] = new_global
            pair_to_words.setdefault(pair, set()).add(word_id)
            # Stage 4: 如果提供了堆，则推入新条目
            if heap is not None and vocab is not None:
                push_heap(heap, pair, new_global, vocab)

    # 5. 更新 word 对象
    word.tokens = new_tokens
    word.pair_counts = new_local
