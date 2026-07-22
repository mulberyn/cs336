# BPE 训练测试报告

> **日期:** 2026-07-22
> **项目:** CS336 Assignment 1 — BPE Tokenizer

---

## 1. 验证集测试结果

**命令:**

```bash
uv run python -c "
from cs336_basics.tokenizer.train import train_bpe
vocab, merges = train_bpe(
    input_path='data/TinyStoriesV2-GPT4-valid.txt',
    vocab_size=5000,
    special_tokens=['<|endoftext|>'],
    output_dir='out/tokenizer',
)
"
```

**测试配置:**

| 参数         | 值                                  |
| ------------ | ----------------------------------- | --------- | --- |
| 输入文件     | `data/TinyStoriesV2-GPT4-valid.txt` |
| 文件大小     | 22 MB                               |
| 目标词表大小 | 5000                                |
| 特殊 token   | `<                                  | endoftext | >`  |
| 输出目录     | `out/tokenizer`                     |

**测试结果:**

```
Training BPE: 100%|██████████| 5000/5000 [00:00<00:00, 5258.28token/s]
Tokenizer saved to .../out/tokenizer

====================================================
Phase                       Time (s)        %
----------------------------------------------------
load_text                       0.08     2.1%
pretokenize+count               2.47    69.3%
build_structures                0.05     1.3%
training_loop                   0.92    25.7%
save                            0.06     1.6%
----------------------------------------------------
TOTAL                           3.57
Merges performed                4743
====================================================
Vocabulary size: 5000
Merge rules:     4743
```

**输出文件:**

| 文件                       | 大小   | 说明                      |
| -------------------------- | ------ | ------------------------- |
| `out/tokenizer/vocab.json` | 118 KB | 5000 个 token（hex 编码） |
| `out/tokenizer/merges.txt` | 60 KB  | 4743 条合并规则           |

**结论:** 在 22 MB 验证集上，总计 **3.57 s** 完成训练。预分词占 69.3% 为最大瓶颈，这与[优化分析报告](./optimization-analysis.md)的结论一致。

---

## 2. 完整训练集预估

**命令:**

```bash
uv run python -m cs336_basics.tokenizer.train
```

该命令等价于：

```python
train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",  # 2.2 GB
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    output_dir="out/tokenizer",
)
```

**与验证集测试的关键差异:**

| 参数         | 验证集 (已测)     | 训练集 (预估)      | 倍数      |
| ------------ | ----------------- | ------------------ | --------- |
| 文件大小     | 22 MB             | 2,228 MB           | **~100×** |
| 目标词表大小 | 5,000             | 10,000             | **2×**    |
| 预分词策略   | `findall`（快速） | `finditer`（流式） | —         |
| 预计合并次数 | 4,743             | ~9,743             | **~2×**   |

**预估耗时（外推）:**

```
====================================================
Phase                       Est. Time (s)        %
----------------------------------------------------
load_text                       ~3–5         0.5%
pretokenize+count             ~300–400      90–95%
build_structures                 ~3–5        < 1%
training_loop                   ~10–15        2–3%
save                             ~1          < 1%
----------------------------------------------------
TOTAL                         ~320–420
Merges performed               ~9,743
====================================================
```

即总计约 **5–7 分钟**，其中预分词阶段占 90% 以上。

> **注意:** 训练集 2.2 GB 超过自适应阈值（50 MB），因此预分词使用流式 `finditer`
> 而非 `findall`，比验证集的快速路径慢约 25%。同时，训练集的唯一词类型更多
> （预估 5 万+，验证集约 1.7 万），训练循环中每次合并需要更新更多词，因此
> 训练循环实际耗时会比简单 ×2 更高。

**`tqdm` 进度条预期表现:**

```
Training BPE:   3%|▎         | 257/10000 [00:00<05:00, 32.45token/s]
Training BPE:   5%|▌         | 500/10000 [00:15<04:45, 33.30token/s]
Training BPE:  10%|█         | 1000/10000 [00:30<04:30, 33.28token/s]
...
Training BPE: 100%|██████████| 10000/10000 [05:20<00:00, 31.25token/s]
```

初始的 257 个 token（256 单字节 + 1 特殊 token）已存在，所以进度从 3% 开始。
每个 merge 创建一个新 token，进度条逐步走到 100%。

---

## 3. 快速验证命令（推荐）

开发过程中推荐使用以下命令进行快速验证，避免每次等待 5–7 分钟：

```bash
# 快速验证：验证集 + 小词表（~3 秒）
uv run python -c "
from cs336_basics.tokenizer.train import train_bpe
vocab, merges = train_bpe(
    input_path='data/TinyStoriesV2-GPT4-valid.txt',
    vocab_size=1000,
    special_tokens=['<|endoftext|>'],
)
"

# 单元测试（~1.5 秒）
uv run pytest tests/test_train_bpe.py -v

# 完整训练（~5–7 分钟，仅最终验证时运行）
uv run python -m cs336_basics.tokenizer.train
```

---

## 4. 硬件建议

| 配置                 | 训练集耗时（预估） | 说明                                            |
| -------------------- | ------------------ | ----------------------------------------------- |
| 16 GB RAM, Apple M1+ | ~5–7 min           | 内存充足，不会触发 swap                         |
| 8 GB RAM             | ~10–15 min         | 2.2 GB 字符串 + 数据结构可能触发 swap，显著拖慢 |
| 4 GB RAM             | 不建议             | 可能 OOM；请使用验证集测试                      |

预分词是纯 CPU 计算密集型任务，**不受 GPU 加速**。多核 CPU 可以利用后续
"分块并行预分词"优化（见[优化分析报告](./optimization-analysis.md)第 4.1 节）。
