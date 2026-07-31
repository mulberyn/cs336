# 实现指南：`train_bpe.py`

## 概述

一个 CLI 脚本，使用 `cs336_basics.tokenizer` 包在原始文本语料库上训练 BPE tokenizer，并将结果持久化到磁盘。

## 职责

- 解析命令行参数（输入文件、词表大小、特殊 token、输出目录、worker 数量）。
- 使用解析后的参数调用 `cs336_basics.tokenizer` 中的 `train_bpe()`。
- 优雅地处理错误（输入文件缺失、词表大小无效、磁盘空间不足）。
- 向用户报告训练进度和耗时。
- 失败时以非零状态码退出。

## 推荐结构

### 入口点（`main()` 或 `if __name__ == "__main__"`）

1. **解析参数** — 使用 `argparse.ArgumentParser`。
2. **验证输入** — 检查输入路径存在且为可读文件；验证 `vocab_size >= 256`。
3. **运行训练** — 调用 `train_bpe(...)` 并捕获已知异常。
4. **报告结果** — 打印输出路径和简要总结。
5. **退出** — 成功时 `sys.exit(0)`，失败时 `sys.exit(1)`。

## CLI 参数

| 参数 | 类型 | 必需 | 描述 |
|---|---|---|---|
| `input_path` | `str`（位置参数） | 是 | UTF-8 训练语料库路径 |
| `--vocab-size` | `int` | 是 | 目标词表大小（≥ 256） |
| `--special-tokens` | `str` (nargs="*") | 否 | 特殊 token（例如 `<\|endoftext\|>`） |
| `--output-dir` | `str` | 否 | vocab.json / merges.txt 的输出目录 |
| `--num-workers` | `int` | 否 | 并行预分词 worker 数 |

## 集成点

- **`cs336_basics.tokenizer.train_bpe(input_path, vocab_size, special_tokens, output_dir, num_workers)`**
  - 返回 `(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]])`
  - 内部已处理计时、进度条和序列化。
- **`cs336_basics.tokenizer.save_tokenizer(vocab, merges, output_dir)`**
  - 写入 `vocab.json`（十六进制编码）和 `merges.txt`（十六进制编码，每行一个 pair）。

## 边界情况与错误处理

- **输入文件缺失/不可读** → 捕获 `FileNotFoundError` / `PermissionError`，打印描述性消息，退出代码 1。
- **词表大小过小**（< 256 + len(special_tokens)）→ 打印错误并在调用 `train_bpe` 前退出。
- **空输入文件** → `train_bpe` 会产生仅包含单字节 token 和特殊 token 的词表；合并列表为空。打印警告但不崩溃。
- **输出目录不可写** → 捕获 `save_tokenizer` 中的 `OSError`。
- **KeyboardInterrupt** → 让信号传播或打印一条干净的消息后退出。

## 行业标准

- 使用 `if __name__ == "__main__":` 保护。
- 将进度和计时信息打印到 stderr，最终结果打印到 stdout。
- 使用 `argparse.RawDescriptionHelpFormatter` 获得清晰的 `--help` 消息。
- 在参数解析器中包含一行 epilog（例如 `"在文本语料库上训练 BPE tokenizer。"`）。
- 成功退出码为 0，任何错误为 1。
- 当训练提前停止时（在达到 vocab_size 之前没有更多可合并的 pair）记录警告。

## 测试关注点

- `train_bpe` 函数由课程测试套件通过 `run_train_bpe` 适配器进行单元测试。
- 此脚本是 CLI 包装器；通过将其作为子进程针对小型测试语料库运行并验证输出文件存在且格式正确来进行测试。
