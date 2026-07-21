from collections import Counter

class Word:
    """表示一个单词的 token 序列及其统计信息"""
    __slots__ = ('tokens', 'freq', 'pair_counts')
    
    tokens: list[int]          # 整数 token ID 列表
    freq: int                  # 该单词在语料中的总频次
    pair_counts: Counter[tuple[int, int]]   # 单词内部相邻 pair 的局部计数

    def __init__(self, tokens: list[int], freq: int):
        self.tokens = tokens
        self.freq = freq
        self.pair_counts = Counter()
        for i in range(len(tokens) - 1):
            self.pair_counts[(tokens[i], tokens[i+1])] += 1