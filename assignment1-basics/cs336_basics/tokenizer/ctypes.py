from collections import Counter


class Word:
    __slots__ = ('tokens', 'freq', 'pair_counts')
    
    # tokens: list[int]  # 例如 [97, 98, 99] 对应 "abc"
    # freq: int
    # pair_counts: Counter[Tuple[int, int]]  # 例如 {(97, 98): 1, (98, 99): 1}
    
    def __init__(self, tokens: list[int], freq: int):
        self.tokens = tokens
        self.freq = freq
        self.pair_counts = Counter()
        for i in range(len(tokens) - 1):
            self.pair_counts[(tokens[i], tokens[i + 1])] += 1