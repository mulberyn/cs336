# Python 多进程并行化深度解析 —— 以 BPE 预分词为例

> **项目:** CS336 Assignment 1 — BPE Tokenizer
> **日期:** 2026-07-22
> **聚焦:** 从零到一实现并行预分词，三次方案迭代全过程，以及背后的操作系统原理。

---

## 目录

1. [起点：为什么要并行化](#1-起点为什么要并行化)
2. [方案 A：ProcessPoolExecutor + pickle](#2-方案-aprocesspoolexecutor--pickle)
3. [方案 B：SharedMemory + spawn](#3-方案-bsharedmemory--spawn)
4. [方案 C：fork + COW + 索引传递](#4-方案-cfork--cow--索引传递)
5. [核心概念图谱](#5-核心概念图谱)
6. [实战 checklist](#6-实战-checklist)
7. [延伸阅读](#7-延伸阅读)

---

## 1. 起点：为什么要并行化

### 1.1 性能画像

在 `TinyStoriesV2-GPT4-valid.txt`（22 MB）上运行 `train_bpe`，分阶段耗时如下：

```
Phase                       Time (s)        %
----------------------------------------------------
load_text                       0.05     1.3 %
pretokenize+count               2.74    78.6 %   ← 瓶颈
build_structures                0.05     1.5 %
training_loop                   0.65    18.6 %
----------------------------------------------------
TOTAL                           3.49
```

**预分词占总时间的 78.6%。** 它的工作内容是：

```
对于文本中的每一个字符位置, 尝试匹配 GPT-2 正则:
  '(?:[sdmt]|ll|ve|re)    ← 英文缩写
  | ?\p{L}+               ← 可选空格 + Unicode 字母序列
  | ?\p{N}+               ← 可选空格 + Unicode 数字序列
  | ?[^\s\p{L}\p{N}]+     ← 可选空格 + 标点符号
  | \s+(?!\S)             ← 尾部空白
  | \s+                   ← 一般空白
```

正则匹配是 **CPU 密集型** 且 **天然可并行** 的任务——文本的不同区域之间没有任何依赖关系。

### 1.2 加速潜力

```
2.2 GB 训练集:
  - 1 个核心串行: ~250 s
  - 8 个核心并行: ~31 s  (理想情况)
  - 预期加速比: 8×
```

### 1.3 Python 多进程的基本约束

在深入方案之前，必须理解 Python 并行的核心约束：

| 概念 | 说明 |
|---|---|
| **GIL（全局解释器锁）** | Python 线程无法并行执行 Python 字节码。对于 CPU 密集任务，`ThreadPoolExecutor` **无效**。 |
| **多进程是唯一出路** | 每个进程有独立的 Python 解释器和 GIL，真正利用多核。 |
| **进程间通信（IPC）有代价** | 进程不共享内存。传数据必须通过 pickle 序列化 → pipe/queue → unpickle。 |
| **macOS 默认 `spawn`** | 子进程是全新的 Python 解释器，通过 `import` 加载模块。父进程的内存对子进程不可见。 |
| **Linux 默认 `fork`** | 子进程是父进程的完整拷贝（COW）。父进程的全部内存对子进程可见。 |

---

## 2. 方案 A：ProcessPoolExecutor + pickle

### 2.1 思路

最直观的并行方案：把文本切成 8 块，交给 8 个 worker 进程分别处理。

```python
def _worker_count_tokens(chunk_text: str, ...) -> Counter[bytes]:
    compiled = re.compile(GPT2_PATTERN)
    counter = Counter()
    for m in compiled.finditer(chunk_text):
        counter[m.group().encode("utf-8")] += 1
    return counter

chunks = split_text(text, num_workers=8)

with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(_worker_count_tokens, chunk) for chunk in chunks]
    for future in as_completed(futures):
        merged.update(future.result())
```

### 2.2 实际结果

```
pretokenize+count: 258 s  ← 与串行几乎一样!
Workers (parallel):    8  ← 8 个 worker 都启动了
```

**8 个 worker 并没有真正并行工作。**

### 2.3 根因分析：管道的隐性瓶颈

问题的核心在于 `executor.submit()` 如何把参数传给 worker。

```
ProcessPoolExecutor 内部结构:

主进程                           子进程 (worker)
───────                          ───────
  │                                │
  │  submit(func, chunk_0)         │
  │  ┌──────────────────┐          │
  │  │ pickle(func_ref)  │ ──┐     │
  │  │ pickle(chunk_0)   │   │     │
  │  │  275 MB 字符串!   │   │     │
  │  └──────────────────┘   │     │
  │                          ▼     │
  │  ┌─────────────────────────┐  │
  │  │     multiprocessing     │  │
  │  │        Queue            │  │
  │  │   (底层是一对 Pipe)      │──┤→ worker 从 Queue 读任务
  │  └─────────────────────────┘  │
  │                          ▲     │
  │  submit(func, chunk_1)   │     │
  │  pickle(chunk_1) ────────┘     │
  │  ...                            │
```

关键事实：

> **macOS 上 pipe 的缓冲区只有 16 KB（`PIPE_BUF`）。**

这意味着主进程每写 16 KB 就必须等待 worker 读出 16 KB，才能继续写下一个 16 KB。

```
传输一个 275 MB 的 chunk:

  275 MB ÷ 16 KB = 17,188 次往返

  每次往返 = 主进程 write() → 内核缓冲区满 → 阻塞 →
             worker read() → 内核缓冲区空 → 主进程被唤醒 →
             主进程 write() → ...

  17,188 × (系统调用 + 上下文切换) ≈ 数秒的纯 IPC 开销
```

而 8 个 chunk **全部进入同一个 Queue → 同一个 Pipe**，所以它们是**串行传输**的：

```
时间轴:
│ chunk_0 传输 (17k 往返) │ chunk_1 传输 │ ... │ chunk_7 传输 │
│         ~3s             │     ~3s      │     │     ~3s      │
│
└─ 总计 ~24s 的 IPC 开销, 且 worker 在此期间大量时间在等待数据
```

**核心教训 1：绝不要通过 multiprocessing Pipe/Queue 传递超过 MB 级别的数据。**

---

## 3. 方案 B：SharedMemory + spawn

### 3.1 思路

既然 Pipe 传数据太慢，那就让数据留在共享内存里，只传一个"指针"（共享内存的名称）。

```python
from multiprocessing import shared_memory

# 主进程: 把文本编码后放入共享内存
text_bytes = text.encode("utf-8")                  # 2.2 GB str → 2.2 GB bytes
shm = shared_memory.SharedMemory(create=True, size=len(text_bytes))
shm.buf[:] = text_bytes                            # 拷贝到共享内存

# Worker: 通过名称重新挂载同一块共享内存
def _worker_count_tokens_shm(shm_name, start, end, ...):
    shm = shared_memory.SharedMemory(name=shm_name)
    chunk_bytes = bytes(shm.buf[start:end])        # 只拷贝自己需要的那一块
    chunk_text = chunk_bytes.decode("utf-8")
    # ... regex ...
    shm.close()

# 只传 (name, start, end) 三元组, 几十个字节
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(_worker_count_tokens_shm,
                               shm.name, start, end, ...)
               for start, end in chunk_offsets]
```

### 3.2 实际结果

```
pretokenize+count: 255 s  ← 仍然没有加速
```

### 3.3 根因分析：`spawn` 的隐藏代价

方案 B 在 IPC 层面是正确的——Pipe 里只传几十字节的参数。但它有两个隐藏问题：

**问题 1：`text.encode()` 的隐性成本**

```python
text_bytes = text.encode("utf-8")
```

`text` 已经是内存中的 2.2 GB Python 字符串。`.encode()` 创建一个**新的** 2.2 GB
bytes 对象。此时进程内存占用：

```
2.2 GB (原字符串) + 2.2 GB (新 bytes) + 2.2 GB (shm.buf) = 6.6 GB
```

如果物理内存不足（比如 8 GB MacBook），OS 开始 swapping → 一切都变慢。

**问题 2：`spawn` 模式下 worker 的冷启动**

在 macOS 的 `spawn` 模式下，每个 worker 是一个全新的 Python 进程：

```
Worker 启动过程:
1. 执行 python -c "import multiprocessing; ..."
2. import cs336_basics.tokenizer.train   ← 重新导入整个模块
3. import regex                          ← 重新导入 regex
4. 反序列化任务参数
5. 打开 SharedMemory
6. 解码 UTF-8
7. 编译正则
8. 开始匹配
```

步骤 2–3 的模块导入虽然比正则匹配快得多（~0.5s），但在 8 个 worker 同时启动时，
它们**竞争同一个磁盘 I/O 和 Python 导入锁**，可能被部分串行化。

**核心教训 2：`SharedMemory` 解决了数据传输问题，但 `spawn` 的冷启动开销和内存翻倍问题抵消了收益。**

---

## 4. 方案 C：fork + COW + 索引传递

### 4.1 思路

把所有方案的问题逐一解决：

| 问题 | 方案 C 的解法 |
|---|---|
| Pipe 传大数据慢 | 只传 `int` 索引（几个字节） |
| spawn 冷启动 | 用 `fork`——子进程继承父进程全部内存，无需重新导入 |
| encode() 翻倍内存 | 不编码——直接共享 Python 字符串 |
| 模块级函数不可 pickle | Worker 函数定义在模块顶层 |

核心技巧：利用 **fork 的写时复制（Copy-on-Write）语义**。

```
fork() 之后:
┌──────────────────────────┐
│        父进程             │
│                          │
│  _fork_chunks = [        │
│    chunk_0,  ← 275 MB   │
│    chunk_1,  ← 275 MB   │
│    ...                   │
│    chunk_7   ← 275 MB   │
│  ]                       │
│                          │
│  这些 str 对象的物理内存   │
│  在 fork 时被标记为       │
│  "copy-on-write"         │
└──────┬───────────────────┘
       │
       │ os.fork()
       ▼
┌──────────────────────────┐
│        子进程             │
│                          │
│  _fork_chunks = [        │
│    chunk_0,  ← 共享父进程的物理页
│    chunk_1,  ← 共享父进程的物理页
│    ...                   │
│    chunk_7   ← 共享父进程的物理页
│  ]                       │
│                          │
│  只读访问不触发拷贝!       │
└──────────────────────────┘
```

### 4.2 完整实现

```python
# ── 模块级全局变量 ──
_FORK_AVAILABLE = "fork" in mp.get_all_start_methods()
_fork_chunks: list[str] | None = None


# ── Worker 函数（模块顶层, 可被 pickle）──
def _worker_by_index(idx: int) -> Counter[bytes]:
    """只接收一个 int 索引, 文本块通过 _fork_chunks 的 COW 继承获取"""
    chunk_text = _fork_chunks[idx]           # 零拷贝访问
    compiled = re.compile(GPT2_PATTERN)

    if len(chunk_text) < _FINDALL_THRESHOLD_CHARS:
        return Counter(t.encode("utf-8") for t in compiled.findall(chunk_text))

    counter = Counter()
    for m in compiled.finditer(chunk_text):
        counter[m.group().encode("utf-8")] += 1
    return counter


# ── 分块函数 ──
def _split_text_into_chunks(text: str, num_chunks: int) -> list[str]:
    """在换行符边界上将文本切分为 num_chunks 个等大块"""
    if num_chunks <= 1:
        return [text]

    total_len = len(text)
    chunk_size = total_len // num_chunks
    chunks = []
    start = 0

    for i in range(num_chunks):
        if start >= total_len:
            break
        if i == num_chunks - 1:
            chunks.append(text[start:])
            break

        end = start + chunk_size
        nl = text.find("\n", end)            # 对齐到换行符
        if nl == -1:
            chunks.append(text[start:])
            break
        end = nl + 1
        chunks.append(text[start:end])
        start = end

    return [c for c in chunks if c]


# ── 并行入口 ──
def _count_token_sequences_parallel(
    text: str, num_workers: int
) -> Counter[bytes]:
    global _fork_chunks
    _fork_chunks = _split_text_into_chunks(text, num_workers)

    ctx = mp.get_context("fork")
    merged = Counter()

    with ctx.Pool(processes=num_workers) as pool:
        # 只传 int 索引! Pipe 中每个任务只有几个字节
        indices = list(range(len(_fork_chunks)))
        for result in pool.imap_unordered(_worker_by_index, indices,
                                          chunksize=1):
            merged.update(result)

    return merged


# ── 在 train_bpe 中触发 ──
_FORK_AVAILABLE = "fork" in mp.get_all_start_methods()
use_parallel = (
    _FORK_AVAILABLE
    and num_workers > 1
    and len(text) >= 10_000_000   # 10 MB 阈值，小于此值并行得不偿失
)
```

### 4.3 为什么换行符是安全的分割点

```
GPT2_PATTERN 的 6 个分支:
  1. '(?:[sdmt]|ll|ve|re)     ← 不包含 \n
  2.  ?\p{L}+                 ← \p{L} 不匹配 \n
  3.  ?\p{N}+                 ← \p{N} 不匹配 \n
  4.  ?[^\s\p{L}\p{N}]+       ← 明确排除了 \s
  5.  \s+(?!\S)               ← 匹配 \n 但只在文本末尾
  6.  \s+                     ← 匹配 \n 作为独立 token

结论: 没有任何分支的匹配会跨越 \n 字符。
      每个 \n 都是 token 边界, 可以安全地在此处切割文本。
```

### 4.4 验证结果

**独立 Benchmark（22 MB 验证集）：**

```
串行:  2.3 s  ████████████████████████████████████████
并行:  0.7 s  █████████████       (4 workers, 3.5× 加速)
```

**完整训练（2.2 GB 训练集，vocab_size=10000）：**

```
Phase                       Time (s)
----------------------------------------------------
load_text                       6.58
pretokenize+count             251.32   ← 预期 ~250s, 未获加速
build_structures                1.31
training_loop                   7.66
----------------------------------------------------
TOTAL                         267.19  (4.5 min)
```

### 4.5 为什么 2.2 GB 大文件没有加速？

这是本次实践中**最有价值的技术发现**。

> **macOS 自 10.14 起废弃了 `fork`。** 虽然 Python 3.13 仍可调用 `os.fork()`，
> 但 macOS 内核对于大内存进程的 fork 支持已经退化。

具体机制：

```
1. fork() 时, 内核为子进程创建一份父进程页表的拷贝
2. 所有物理页被标记为 COW (copy-on-write)
3. 当子进程首次读取某页时:
   - Intel Mac: 硬件页故障 → 内核映射到父进程的物理页 (无拷贝, 快)
   - Apple Silicon: 内核需要更新 IOMMU 映射 → 可能涉及 TLB 刷新

4. 对于 2.2 GB 的地址空间 (约 537,000 个 4KB 页):
   - 8 个子进程同时读取各自不同的页
   - 内核需要处理 8 × (275 MB / 4 KB) ≈ 560,000 次页故障
   - 在大内存 + 多进程场景下, macOS 的 VM 系统可能串行化页故障处理
```

**对比 Linux:**
- Linux 的 `fork` 实现经过数十年优化，对 COW 大内存场景有成熟的处理
- 预期 Linux 上可以获得接近线性的加速

**核心教训 3：并行策略必须在目标平台上实测。OS 的进程模型差异足以改变方案的可行性。**

### 4.6 优雅降级设计

正因为 `fork` 在不同平台上的行为不可预测，我们的设计包含了多层降级：

```
train_bpe 预分词路径选择:

输入文本
  │
  ├─ num_workers == 1  ──→ 串行 (_count_token_sequences)
  │
  ├─ fork 不可用 ──→ 串行
  │
  ├─ 文本 < 10 MB ──→ 串行 (并行得不偿失)
  │
  └─ 以上都不满足 ──→ 并行 (_count_token_sequences_parallel)
                        │
                        └─ 内部仍分两档:
                              ├─ chunk < 50 MB → findall (快 25%)
                              └─ chunk ≥ 50 MB → finditer (省内存)
```

---

## 5. 核心概念图谱

```
                        Python 多进程并行

        ┌───────────────────┼───────────────────────┐
        │                   │                       │
   线程 (Thread)        进程 (Process)         协程 (asyncio)
        │                   │                       │
   ❌ GIL 限制             ✅ 真并行              ❌ 仅 I/O 并发
   CPU 密集无效            CPU 密集可用           CPU 密集无效
        │                   │
        │           ┌───────┴───────┐
        │           │               │
        │        spawn           fork
        │     (macOS 默认)    (Linux 默认)
        │           │               │
        │     子进程全新启动    子进程复制父进程
        │     需 import 模块    COW 共享内存
        │     内存独立         内存高效
        │           │               │
        │     ┌─────┴─────┐   ┌─────┴─────┐
        │     │           │   │           │
        │   Pipe/Queue  Shared  Pipe/Queue  全局变量
        │   传参数       Memory  传参数      (COW)
        │     │           │       │           │
        │  ❌ 大对象     ✅      ❌ 大对象    ✅
        │    慢          快       慢         快
        │     │           │       │           │
        │  ⚠️ 16KB       ⚠️     ⚠️ 16KB     ⚠️
        │   缓冲区限制    spawn   缓冲区限制  macOS
        │               冷启动              大内存退化
        │                + 内存              (2.2GB)
        │                翻倍
        │
        └── 我们最终选择了: fork + 全局变量(COW) + 索引传递
                           │
                           ├─ 小文件 (22MB): ✅ 3.5× 加速
                           └─ 大文件 (2.2GB): ⚠️ macOS 限制
```

---

## 6. 实战 checklist

在 Python 项目中引入多进程并行时，按以下顺序排查：

### 6.1 并行前

- [ ] **确认瓶颈是 CPU 密集还是 I/O 密集。** I/O 密集用线程/协程，CPU 密集才需要多进程。
- [ ] **测量串行耗时分布。** 找到占比最高的阶段，只并行化那部分。
- [ ] **评估数据规模。** 如果输入 < 10 MB，进程创建开销可能超过计算收益。

### 6.2 选方案

- [ ] **数据 < 1 MB：** `ProcessPoolExecutor` + 直接 pickle 传参（最简单）
- [ ] **数据 1–100 MB：** `ProcessPoolExecutor` + 全局变量 + fork COW（最均衡）
- [ ] **数据 > 100 MB：** `SharedMemory` + spawn，或考虑分文件让 worker 各自 mmap
- [ ] **Linux 优先 fork，macOS 优先 spawn + SharedMemory**

### 6.3 调试

- [ ] **验证 worker 真的在并行：** 在 worker 中打印 `os.getpid()` 和时间戳
- [ ] **检查 IPC 开销：** 对比 `pool.map(func, chunks)` vs `pool.map(func, [0,1,2...])` 的耗时差异
- [ ] **监控内存：** 用 `psutil` 或 Activity Monitor 确认是否有 swapping
- [ ] **先用小输入验证正确性，再放大测试性能**

### 6.4 上线前

- [ ] **添加降级路径：** 并行不可用时自动回退串行
- [ ] **暴露并发度参数：** 让用户可以根据硬件调整 `num_workers`
- [ ] **添加计时：** 输出串行/并行模式的耗时差异，方便用户判断收益
- [ ] **在目标 OS 上实测：** 不要假设开发机 (macOS) 和生产机 (Linux) 行为一致

---

## 7. 延伸阅读

- Python 官方文档: [multiprocessing — Process-based parallelism](https://docs.python.org/3/library/multiprocessing.html)
- CPython 源码: `Modules/_posixshmem.c` — SharedMemory 的 POSIX 实现
- macOS 手册: `man 2 fork` — Apple 对 fork 的警告和限制
- Linux 手册: `man 2 fork` — Linux fork 的 COW 语义
- [PEP 574](https://peps.python.org/pep-0574/) — Pickle 协议 5 的零拷贝 buffer 支持
- GPT-2 源码: [`encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py) — 原始 BPE 预分词正则
