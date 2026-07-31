import mmap
import heapq
from collections import Counter
from pathlib import Path
import regex as re
from tqdm import tqdm
from collections import defaultdict

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def load_text(
    path: str | Path
) -> str:
    path = Path(path)
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.read().decode("utf-8")


def pretokenize(
    text: str,
    special_tokens: list[str] | None = None,
) -> list[str]:
    if not special_tokens:
        return re.findall(GPT2_PATTERN, text)
    escaped = "|".join(re.escape(tok) for tok in special_tokens)
    parts = re.split(f"({escaped})", text)
    pretokens: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            pretokens.append(part)
        else:
            pretokens.extend(re.findall(GPT2_PATTERN, part))
    return pretokens


def initialize_vocab(
    special_tokens: list[str],
) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab


class ReversePairKey:
    """
    用于处理频率相同时，取出字典序最大的 pair
    """
    __slots__ = ("freq", "p0", "p1", "v0", "v1")
    def __init__(self, freq, p0, p1, v0, v1):
        self.freq = freq
        self.p0 = p0
        self.p1 = p1
        self.v0 = v0
        self.v1 = v1


    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq > other.freq
        return (self.v0, self.v1) > (other.v0, other.v1)


def update(idx, p0, p1, new_id, words, pair_freq, pair_to_indices, heap, vocab):
    old_word = words[idx]
    new_word = []
    i = 0
    while i < len(old_word):
        # 相邻两个是要合并的
        if i + 1 < len(old_word) and old_word[i] == p0 and old_word[i + 1] == p1:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(old_word[i])
            i += 1
    changed_pairs = set()
    # 更新旧的 freq
    for i in range(len(old_word) - 1):
        pair = (old_word[i], old_word[i + 1])
        pair_freq[pair] -= 1
        changed_pairs.add(pair)
        if pair_freq[pair] == 0:
            del pair_freq[pair]
        pair_to_indices[pair].discard(idx)
        if not pair_to_indices[pair]:
            del pair_to_indices[pair]
    # 更新新的 freq
    for i in range(len(new_word) - 1):
        pair = (new_word[i], new_word[i + 1])
        changed_pairs.add(pair)
        pair_freq[pair] += 1
        pair_to_indices[pair].add(idx)
    
    for pair in changed_pairs:
        heapq.heappush(heap, ReversePairKey(pair_freq[pair], pair[0], pair[1], vocab[pair[0]], vocab[pair[1]]))
    
    words[idx] = new_word


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    text = load_text(input_path)
    pretokens = pretokenize(text, special_tokens) # pretokens: list[str]
    vocab = initialize_vocab(special_tokens)
    
    words = [] # words: list[lists[int]]
    pair_freq = Counter()
    pair_to_indices = defaultdict(set)
    for part in pretokens:
        if part in special_tokens:
            continue
        idx = len(words)
        word = list(part.encode('utf-8')) # word: list[int]
        words.append(word)
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += 1
            pair_to_indices[pair].add(idx)
    
    # 需要注意处理在相同频率下，取字典序较大的
    heap = [ReversePairKey(freq, p0, p1, vocab[p0], vocab[p1]) for (p0, p1), freq in pair_freq.items()]
    heapq.heapify(heap)
    merges = [] # merges: list[bytes, bytes]
    
    with tqdm(total=vocab_size - len(vocab), desc="Training BPE", unit="merge") as pbar:
        while len(vocab) < vocab_size:
            top = heapq.heappop(heap)
            freq, p0, p1 = top.freq, top.p0, top.p1
            # 如果发现和实际统计不同（已经软修改过），则跳过
            if freq != pair_freq.get((p0, p1), 0):
                continue
            new_id = len(vocab)
            vocab[new_id] = vocab[p0] + vocab[p1]
            merges.append((vocab[p0], vocab[p1]))
            # affected_indices 收到影响的 words 索引
            affected_indices = pair_to_indices.pop((p0, p1), )
            for idx in affected_indices:
                update(idx, p0, p1, new_id, words, pair_freq, pair_to_indices, heap, vocab)
            pbar.update(1)
            pbar.set_postfix({"vocab_size": len(vocab)})
    
    return vocab, merges


import base64
import json


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
    vocab_filepath: str,
    merges_filepath: str
) -> None:
    # 将 vocab 的键转为字符串，值转为 Base64 字符串
    vocab_b64 = {str(k): base64.b64encode(v).decode('ascii') for k, v in vocab.items()}
    # 将 merges 的每个 bytes 对转为 Base64 字符串对
    merges_b64 = [
        (base64.b64encode(left).decode('ascii'), base64.b64encode(right).decode('ascii'))
        for left, right in merges
    ]

    vocab_data = {
        "special_tokens": special_tokens,
        "vocab": vocab_b64
    }
    merges_data = {"merges": merges_b64}

    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)   # indent 可选，便于阅读
    with open(merges_filepath, "w", encoding="utf-8") as f:
        json.dump(merges_data, f, indent=2)


def load_tokenizer(
    vocab_filepath: str,
    merges_filepath: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], list[str]]:
    """
    从两个 JSON 文件加载 tokenizer 数据，返回 (vocab, merges, special_tokens)。
    """
    with open(vocab_filepath, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    with open(merges_filepath, "r", encoding="utf-8") as f:
        merges_data = json.load(f)

    # 还原 vocab：键从字符串转回 int，值从 Base64 解码为 bytes
    vocab = {
        int(k): base64.b64decode(v.encode('ascii'))
        for k, v in vocab_data["vocab"].items()
    }
    # 还原 merges：每个 Base64 字符串对解码为 bytes 对
    merges = [
        (base64.b64decode(left.encode('ascii')), base64.b64decode(right.encode('ascii')))
        for left, right in merges_data["merges"]
    ]
    special_tokens = vocab_data.get("special_tokens", []) 

    return vocab, merges, special_tokens