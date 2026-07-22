# BPE Tokenizer 重构与优化 — 学习文档

> **项目:** CS336 Assignment 1 — BPE Tokenizer
> **日期:** 2026-07-22
> **主旨:** 记录从原始 7 文件结构重构为 4 文件工业标准结构，以及并行预分词优化的完整过程。

---

## 目录

1. [重构目标](#1-重构目标)
2. [文件结构对比](#2-文件结构对比)
3. [逐模块对比分析](#3-逐模块对比分析)
   - [3.1 `__init__.py` —— 包入口](#31-__init__py--包入口)
   - [3.2 `core.py` —— 数据结构](#32-corepy--数据结构)
   - [3.3 `bpe.py` —— 算法核心](#33-bpepy--算法核心)
   - [3.4 `train.py` —— 训练流水线](#34-trainpy--训练流水线)
4. [关键优化模式](#4-关键优化模式)
5. [并行预分词深度解析](#5-并行预分词深度解析)
6. [经验总结](#6-经验总结)

---

## 1. 重构目标

| 目标 | 说明 |
|---|---|
| **减少文件碎片** | 7 个文件中有 3 个只含单个函数（`vocabulary.py`、`utils.py`、`pretokenize.py`），过度拆分降低可读性 |
| **统一注释语言** | 原始代码混用中英文注释，统一为英文 docstring（Google 风格） |
| **增强可拓展性** | 清晰的模块边界：数据结构 / 算法 / 训练流程 / 序列化 |
| **添加进度可视化** | 集成 tqdm 进度条 |
| **性能优化** | 流式预处理、自适应策略、并行预分词 |
| **规范序列化** | hex 编码的词表与合并规则持久化 |

---

## 2. 文件结构对比

```
Before (7 files)                    After (4 files)
─────────────────────────────       ─────────────────────────────
tokenizer/                          tokenizer/
├── __init__.py     (5 lines)       ├── __init__.py     (36 lines)  ← 完整包文档
├── core.py         (16 lines)      ├── core.py         (39 lines)  ← 英文 docstring
├── bpe.py          (130 lines)     ├── bpe.py          (235 lines) ← 英文 docstring
├── train.py        (65 lines)      └── train.py        (475 lines) ← 合并 + 新功能
├── vocabulary.py   (9 lines)  ✗
├── utils.py        (10 lines) ✗
└── pretokenize.py  (24 lines) ✗
```

**合并逻辑：**

| 被删除文件 | 合并到 | 原因 |
|---|---|---|
| `vocabulary.py` | `train.py` | 仅 9 行，只被 `train_bpe` 调用 |
| `utils.py` | `train.py` | 仅 10 行，只被 `train_bpe` 调用 |
| `pretokenize.py` | `train.py` | 虽可独立使用，但 24 行单独成文件过轻；保留公开 API |

**保留独立文件的原则：**

- `core.py` — `Word` 类是被 `bpe.py` 和 `train.py` 共同依赖的数据结构
- `bpe.py` — 纯算法实现，与训练流程解耦，可被其他训练策略复用
- `train.py` — 训练流水线 + I/O + 序列化，是唯一的"业务流程"文件

---

## 3. 逐模块对比分析

### 3.1 `__init__.py` —— 包入口

**Before:**
```python
from .train import train_bpe
from .vocabulary import initialize_vocab
from .pretokenize import pretokenize
from .core import Word

__all__ = ['train_bpe', 'initialize_vocab', 'pretokenize', 'Word']
```

**After:**
```python
"""CS336 BPE tokenizer package.

Provides a complete BPE tokenizer training pipeline:

* Corpus loading and GPT-2-style pre-tokenization.
* Incremental BPE training with a heap-based priority queue.
* Serialization of trained vocabularies and merge tables to disk.

Example usage::

    from cs336_basics.tokenizer import train_bpe

    vocab, merges = train_bpe(
        input_path="data/corpus.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        output_dir="out/tokenizer",
    )
"""

from .core import Word
from .train import (
    initialize_vocab,
    pretokenize,
    save_merges,
    save_tokenizer,
    save_vocab,
    train_bpe,
)

__all__ = [
    "train_bpe",
    "initialize_vocab",
    "pretokenize",
    "save_vocab",
    "save_merges",
    "save_tokenizer",
    "Word",
]
```

**变化点：**
- 添加模块级 docstring，包含使用示例
- 导入路径统一为 `from .train import ...`（不再分散在 4 个模块）
- 新增 3 个序列化函数的公开导出

---

### 3.2 `core.py` —— 数据结构

**Before (中文注释):**
```python
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
```

**After (英文 docstring，Google 风格):**
```python
"""Core data structures for the BPE tokenizer."""

from collections import Counter


class Word:
    """Represents a word as a sequence of token IDs with frequency and
    local pair-count statistics.

    Each ``Word`` tracks the token sequence (as integer IDs), the word's
    frequency in the corpus, and the counts of every adjacent token pair
    that appears inside the word.  These local pair counts are used during
    incremental BPE training to efficiently update global pair statistics
    after a merge.

    Attributes:
        tokens: List of integer token IDs that make up this word.
        freq: Total occurrence count of this word in the training corpus.
        pair_counts: Counter mapping each adjacent token pair ``(a, b)``
            to how many times it appears within this specific word.
    """

    __slots__ = ("tokens", "freq", "pair_counts")

    tokens: list[int]
    freq: int
    pair_counts: Counter[tuple[int, int]]

    def __init__(self, tokens: list[int], freq: int) -> None:
        """Initialize a Word with its token sequence and corpus frequency.

        Args:
            tokens: List of integer token IDs for this word.
            freq: Number of times this word appears in the training corpus.
        """
        self.tokens = tokens
        self.freq = freq
        self.pair_counts = Counter()
        for i in range(len(tokens) - 1):
            self.pair_counts[(tokens[i], tokens[i + 1])] += 1
```

**变化点：**
- 模块级 docstring
- 类 docstring 改为 Google 风格，包含详细的用途说明
- `__init__` 方法增加 `Args:` 文档
- 中文字符 `（）` 统一替换为英文
- `__slots__` 使用双引号（风格统一）
- 类型注解保持

---

### 3.3 `bpe.py` —— 算法核心

这是整个 tokenizer 最核心的模块，包含 BPE 增量训练的全部算法逻辑。

**Before 的结构特点:**
- 中文注释混用
- `ReverseBytes` 类的 docstring 写在一行
- `_make_heap_item` 函数体注释使用数字标注（1. 2. 3. 4.）
- `update_word` 函数体使用数字步骤标注（1. 2. 3. 4. 5.）

**After 的结构特点:**
- 模块级 docstring 概述职责
- 所有函数/类使用 Google 风格英文 docstring
- 函数体步骤用 `--- 1. ---` 分隔线标注
- 代码逻辑**完全不变**（保证与参考实现的兼容性）

**核心算法回顾（供学习）：**

```
BPE 增量训练 = 堆 + 倒排索引

数据结构:
  words:           {word_id: Word}          每个词的 token 序列 + 局部 pair 计数
  pair_counts:     {(a,b): global_count}     每个 pair 的全局总频次
  pair_to_words:   {(a,b): {word_id, ...}}   倒排索引：哪些词包含此 pair
  heap:            [(-count, ...)]           大根堆，取最高频 pair

每轮迭代:
  1. pop_best_pair(heap) → 获取当前最高频 pair (a,b)
  2. 创建新 token new_id, 合并词表: vocab[new_id] = vocab[a] + vocab[b]
  3. update_word(每个受影响的词):
     a. 将词中所有 (a,b) 替换为 new_id
     b. 计算局部 pair 计数变化量 (delta)
     c. 将 delta × freq 应用到全局 pair_counts 和 pair_to_words
     d. 推送更新后的 pair 计数入堆
  4. 清理倒排索引中已不存在的 pair

堆的惰性删除:
  当 pair 计数更新时，旧条目不删除，而是推送新条目。
  pop_best_pair 时若发现计数不匹配，跳过旧条目（惰性删除）。
  这比显式删除 O(n) 的堆操作高效得多。
```

**唯一的功能性改动：**

`build_structures` 的入参类型从 `dict[tuple[int, ...], int]` 改为 `dict[bytes, int]`：

```python
# Before: token_sequences 的键是 tuple[int, ...]
# 例如: (104, 101, 108, 108, 111) 表示 "hello"
# 缺点: 每个词需要将 bytes → tuple，O(n) 开销

# After: token_sequences 的键是 bytes
# 例如: b"hello"
# 优点: bytes 天然可哈希，无需转换；内存占用减少约 4×
```

`list(seq)` 对 `bytes` 和 `tuple[int, ...]` 行为一致（都会迭代为 int 列表），
所以 `build_structures` 内部逻辑无需改动。

---

### 3.4 `train.py` —— 训练流水线

这是变化最大的模块。原始 65 行 → 重构后 475 行。

**合并进来的函数：**

| 原始位置 | 函数 | 行数 |
|---|---|---|
| `vocabulary.py` | `initialize_vocab(special_tokens)` | 9 |
| `utils.py` | `load_text(path)` | 10 |
| `pretokenize.py` | `pretokenize(text, special_tokens)` | 24 |

**新增的功能模块：**

```
train.py 结构
├── GPT2_PATTERN            # 预编译正则常量
├── _COMPILED_GPT2          # 预编译正则对象
├── load_text()             # ← 来自 utils.py
├── pretokenize()           # ← 来自 pretokenize.py
├── initialize_vocab()      # ← 来自 vocabulary.py
├── _count_token_sequences()       # 自适应串行计数
├── _fork_chunks / _worker_by_index()  # fork 并行 worker
├── _split_text_into_chunks()      # 换行符对齐分块
├── _count_token_sequences_parallel()  # 并行计数入口
├── save_vocab() / save_merges()   # hex 编码序列化
├── save_tokenizer()               # 一键保存
├── train_bpe()                    # 训练主入口
├── _print_timings()               # 分阶段计时报告
└── __main__                        # CLI demo
```

**`train_bpe` 流程对比：**

```
Before (简版):                          After (完整版):
─────────────────────                   ─────────────────────
1. load_text(path)                      1. load_text(path)
2. pretokenize(text)                    2. initialize_vocab(specials)
3. initialize_vocab(specials)           3. 自适应预分词:
4. build token_sequences                   ├─ 文本 ≥ 10MB 且 fork 可用 → 并行
5. build_structures(...)                   │  (fork + COW, worker 按索引取块)
6. init_heap(...)                          ├─ 文本 < 50MB → findall (快25%)
7. while len(vocab) < target:              └─ 文本 ≥ 50MB → finditer (省内存)
   ├─ pop_best_pair(heap)              4. build_structures(dict(bytes_counter))
   ├─ vocab[new_id] = ...              5. init_heap(...)
   ├─ update_word(...)                 6. tqdm 进度条 + 训练循环
   └─ 清理                              7. 可选: save_tokenizer(output_dir)
                                       8. 打印分阶段计时报告
```

**关键优化点详解：**

#### 优化 1: bytes 键替代 tuple[int, ...]

```python
# Before: 每个 pre-token 需要两次转换
seq = tuple(w.encode('utf-8'))   # bytes → tuple[int, ...]
token_sequences[seq] = token_sequences.get(seq, 0) + 1

# After: bytes 直接作为键
token_sequences[m.group().encode('utf-8')] += 1
```

**效果:** 消除了每个词的 `tuple()` 转换，内存减少约 4× 每键。

#### 优化 2: 流式 finditer + Counter

```python
# Before: 生成巨大的中间列表
words = pretokenize(text)          # list[str], O(总 token 数) 内存
token_sequences = {}
for w in words:                    # 第二次遍历
    seq = tuple(w.encode('utf-8'))
    token_sequences[seq] = token_sequences.get(seq, 0) + 1

# After: 流式生成器直接计数
token_sequences = Counter()
for m in re.finditer(GPT2_PATTERN, text):   # 生成器, O(1) 内存
    token_sequences[m.group().encode('utf-8')] += 1
```

**效果:** 峰值内存从 O(总 token 数) 降至 O(去重类型数)。对于 2.2 GB
训练集，避免了存储约 5 亿个字符串对象。

#### 优化 3: 自适应 findall / finditer

```python
def _count_token_sequences(text, special_bytes):
    if len(text) < 50_000_000:      # 50 MB 阈值
        # findall: C 实现, 快 ~25%, 但生成完整列表
        return Counter(
            token.encode("utf-8")
            for token in _COMPILED_GPT2.findall(text)
            if token.encode("utf-8") not in special_bytes
        )
    # finditer: 流式, 内存安全
    counter = Counter()
    for m in _COMPILED_GPT2.finditer(text):
        seq = m.group().encode("utf-8")
        if seq not in special_bytes:
            counter[seq] += 1
    return counter
```

**效果:** 小而快的文件用 `findall`（~25% 更快），大文件用 `finditer`（内存安全）。

#### 优化 4: 快速子串检查替代正则搜索

```python
# Before: 全文正则扫描 (O(n))
escaped = "|".join(re.escape(tok) for tok in special_tokens)
parts = re.split(f"({escaped})", text)   # 即使没有匹配也扫描全文

# After: 先做快速子串检查
needs_split = any(tok in text for tok in special_tokens)  # O(n) 但 C 实现
if not needs_split:
    token_sequences = _count_token_sequences(text, special_bytes)
else:
    parts = re.split(...)  # 仅在确实需要时拆分
```

**效果:** 对于不含特殊 token 的语料（最常见情况），避免了不必要的全文正则扫描。

#### 优化 5: tqdm 进度条

```python
pbar = tqdm(
    total=vocab_size,
    initial=len(vocab),        # 从 257 开始（256 字节 + 1 特殊）
    desc="Training BPE",
    unit="token",
    dynamic_ncols=True,        # 自适应终端宽度
)

while len(vocab) < vocab_size:
    # ... merge ...
    pbar.update(1)

pbar.close()
```

#### 优化 6: 分阶段计时

```python
====================================================
Phase                       Time (s)        %
----------------------------------------------------
load_text                       6.58     2.5%
pretokenize+count             251.32    94.1%
build_structures                1.31     0.5%
training_loop                   7.66     2.9%
save                            0.31     0.1%
----------------------------------------------------
TOTAL                         267.19
Merges performed                9743
Workers (parallel)                 8
====================================================
```

训练结束后自动打印，无需外部 profiler 即可定位瓶颈。

#### 新增: 序列化

```python
# 保存格式:
# out/tokenizer/vocab.json  → {"0": "00", "1": "01", ..., "5000": "68656c6c6f"}
# out/tokenizer/merges.txt  → 77 6f
#                              68 65
#                              20 61
#                              ...

# 使用 hex 编码确保任意 bytes 内容均可安全序列化
def _bytes_to_repr(b: bytes) -> str:
    return b.hex()          # b"hello" → "68656c6c6f"

def _repr_to_bytes(s: str) -> bytes:
    return bytes.fromhex(s) # "68656c6c6f" → b"hello"
```

---

## 4. 关键优化模式

### 4.1 模式一: 数据结构选型影响全局性能

```
bytes 作为 dict 键  vs  tuple[int, ...] 作为 dict 键
─────────────────      ─────────────────────────
内存: 紧凑二进制        内存: Python int 对象 × N
哈希: bytes.__hash__   哈希: tuple.__hash__ (遍历所有元素)
创建: w.encode()       创建: tuple(bytes_obj) 额外 O(n) 转换
```

**教训:** 在性能敏感的代码路径上，选用 Python 内置的高效类型
（bytes、memoryview、array）替代通用容器类型（tuple、list），
可以带来数量级的内存和速度提升。

### 4.2 模式二: 生成器优于列表

```python
# 反模式: 先构建完整列表再处理
all_items = [expensive_op(x) for x in huge_input]
result = process(all_items)

# 正模式: 边生成边处理
result = process(expensive_op(x) for x in huge_input)
```

**教训:** 当中间结果不需要随机访问时，使用生成器表达式避免
O(n) 的中间列表内存分配。

### 4.3 模式三: 自适应策略

```python
if len(text) < THRESHOLD:
    return fast_path(text)    # 小输入用快速算法
else:
    return safe_path(text)    # 大输入用内存安全的算法
```

**教训:** 不存在对所有输入规模都最优的单一算法。在入口处根据
输入特征选择不同策略，可以在不牺牲大输入安全性的前提下
优化常见的小输入场景。

### 4.4 模式四: 惰性删除替代显式删除

```python
# 反模式: 堆中删除旧条目 O(n)
heap.remove(old_entry)
heapq.heapify(heap)

# 正模式: 推送新条目，pop 时跳过旧条目 O(log n)
heapq.heappush(heap, new_entry)
# ...
while heap:
    entry = heapq.heappop(heap)
    if is_stale(entry):   # 惰性检查
        continue
    return entry
```

**教训:** 当删除频率远低于插入频率时，惰性删除将 O(n) 降为 O(log n)。

---

## 5. 并行预分词深度解析

### 5.1 问题: 为什么需要并行？

```
串行预分词耗时分布 (22 MB 文本):
─────────────────────────────────
regex 匹配:  ████████████████████  ~2.4s  (78%)
其他:        ██                    ~0.7s  (22%)

对 2.2 GB 训练集: ~250s (4 min) 的纯 CPU 计算
```

正则匹配是 CPU 密集型、高度可并行的任务（不同文本区域之间没有依赖关系）。

### 5.2 方案演进

#### 方案 A: ProcessPoolExecutor + pickle 传块 (❌ 无效)

```python
# 每个 chunk (~275 MB) 通过 pipe 序列化
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(worker, chunk) for chunk in chunks]
```

**问题:** macOS pipe 缓冲区仅 ~16 KB。275 MB 需要 ~17,000 次读写往返。
8 个块串行通过单一 pipe → 实际执行变成串行。

#### 方案 B: SharedMemory (❌ 同样慢)

```python
shm = shared_memory.SharedMemory(create=True, size=len(text_bytes))
shm.buf[:] = text_bytes
# Worker 通过 shm.name 重新挂载
```

**问题:** `text.encode("utf-8")` 创建 2.2 GB 副本，内存峰值翻倍。
Worker 启动后的 regex 匹配仍然被串行化（可能与 macOS 的页错误处理有关）。

#### 方案 C: fork + COW + 索引访问 (✅ 小文件 3.5×)

```python
# 1. 主进程: 分块并存入模块级全局变量
_fork_chunks = _split_text_into_chunks(text, num_workers)

# 2. fork: 子进程通过 COW 继承 _fork_chunks
ctx = mp.get_context("fork")

# 3. Pool 只传递整数索引 (几个字节!)
with ctx.Pool(processes=num_workers) as pool:
    results = pool.imap_unordered(_worker_by_index, range(len(_fork_chunks)))

# 4. Worker 内部按索引访问 COW 继承的文本块
def _worker_by_index(idx: int) -> Counter[bytes]:
    chunk_text = _fork_chunks[idx]   # 零拷贝, COW 继承
    return count_tokens(chunk_text)
```

**核心技巧:**
- 文本数据通过 COW 共享，不进入 pipe
- Pipe 只传递 `int`（几个字节），彻底消除 IPC 瓶颈
- `fork` 后子进程的内存是父进程的快照，读操作不触发 COW 拷贝

**为何大文件加速有限：**
macOS 对 `fork` 的支持从 10.14 起已被弃用。对于 2.2 GB 的地址空间，
8 个子进程同时访问可能触发内核的页表管理串行化。Linux 上的性能表现
预计会显著优于 macOS。

### 5.3 架构图

```
┌──────────────────────────────────────────────────┐
│                    主进程                          │
│                                                    │
│  1. load_text() → 2.2 GB str                      │
│  2. _split_text_into_chunks(text, 8)              │
│     → [chunk_0, ..., chunk_7]                    │
│  3. _fork_chunks = chunks   (模块级全局)          │
│                                                    │
│  4. ctx.Pool(processes=8) ── fork() ─────────┐   │
│     pool.imap_unordered(_worker_by_index,     │   │
│                         [0,1,2,3,4,5,6,7])    │   │
│                                               │   │
│  5. merge Counters ←────────── return ────────┤   │
│                                               │   │
└───────────────────────────────────────────────┼───┘
                                                │
          ┌─────────── COW 继承 ────────────────┤
          ▼          ▼          ▼               ▼
    ┌─────────┐ ┌─────────┐ ... ┌─────────┐
    │ Worker 0│ │ Worker 1│     │ Worker 7│
    │         │ │         │     │         │
    │ chunk = │ │ chunk = │     │ chunk = │
    │ _fork_  │ │ _fork_  │     │ _fork_  │
    │ chunks  │ │ chunks  │     │ chunks  │
    │ [0]     │ │ [1]     │     │ [7]     │
    │         │ │         │     │         │
    │ regex → │ │ regex → │     │ regex → │
    │ Counter │ │ Counter │     │ Counter │
    └─────────┘ └─────────┘     └─────────┘
```

---

## 6. 经验总结

### 6.1 文件组织原则

1. **一个文件一个职责领域，不是一个文件一个函数。** `vocabulary.py`（9 行）
   和 `pretokenize.py`（24 行）应该合并到它们所属的业务流程中。

2. **依赖方向决定文件边界。** `Word` 类被 `bpe.py` 和 `train.py` 同时依赖 →
   独立成文件。`initialize_vocab` 只被 `train.py` 调用 → 放在 `train.py`。

3. **公开 API 的粒度应稳定。** 向外暴露的函数（`pretokenize`、`save_vocab` 等）
   即使实现位置变了，导入路径也应保持向后兼容。

### 6.2 性能优化原则

1. **先测量，后优化。** 分阶段计时表明确指出了预分词占 94% 时间，
   让我们把精力集中在正确的方向上。

2. **选择正确的数据类型比微优化代码更重要。** `bytes` 替代 `tuple[int, ...]`
   带来的收益远超任何算法微调。

3. **并行 ≠ 加速。** 在 macOS 上，`fork` 的大内存 COW 行为在 22 MB 时 3.5× 加速，
   但在 2.2 GB 时几乎无效。**并行策略必须在目标硬件上实测。**

4. **自适应策略是性价比最高的优化。** 对于 90% 的小文件场景用快路径，
   10% 的大文件场景用安全路径，比试图为所有场景找一个"最优"算法更实用。

### 6.3 工业级代码风格

1. **docstring 是文档的第一来源。** 每个公开函数都应有 Google 风格的
   docstring，包含简短描述、Args、Returns、使用示例。

2. **常量应有名字和注释。** `_FINDALL_THRESHOLD_CHARS = 50_000_000` 比
   到处写 `50000000` 好理解，也更容易调整。

3. **错误处理要优雅降级。** `fork` 不可用时自动回退串行路径，
   而不是崩溃或报错。

4. **可观测性内建。** 分阶段计时和 tqdm 进度条让用户无需外部工具
   就能理解程序在做什么、哪里慢。

### 6.4 本次重构的完整文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `tokenizer/__init__.py` | 修改 | 包文档 + 统一导入 |
| `tokenizer/core.py` | 修改 | 英文 docstring |
| `tokenizer/bpe.py` | 修改 | 英文 docstring + bytes 键 |
| `tokenizer/train.py` | 重写 | 合并 + 优化 + 序列化 + 并行 |
| `tokenizer/vocabulary.py` | 删除 | → `train.py` |
| `tokenizer/utils.py` | 删除 | → `train.py` |
| `tokenizer/pretokenize.py` | 删除 | → `train.py` |
| `docs/optimization-analysis.md` | 新增 | 瓶颈分析与优化方向 |
| `docs/test-report.md` | 新增 | 测试结果与命令参考 |
| `docs/parallel-pretokenization-report.md` | 新增 | 并行预分词实施报告 |
| `docs/refactoring-study-guide.md` | 新增 | 本文档 |
