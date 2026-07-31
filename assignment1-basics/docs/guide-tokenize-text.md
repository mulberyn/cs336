# 实现指南：`tokenize_text.py`

## 概述

一个 CLI 脚本，从磁盘加载训练好的 BPE tokenizer 并对文本文件进行分词，输出整数 token ID（每行一个序列，空格分隔）。

## 职责

- 从磁盘加载 tokenizer 词表和合并表。
- 读取输入文本文件（按行或整体读取，取决于大小）。
- 使用加载的 `Tokenizer` 对每行或每段进行分词。
- 将生成的 token ID 序列写入输出文件。
- 使用流式处理处理大文件以避免内存压力。
- 为常见失败模式提供清晰的错误消息。

## 推荐结构

### 入口点（`main()`）

1. **解析参数** — `argparse.ArgumentParser`。
2. **验证输入** — 检查词表文件、合并文件和输入文本文件存在。
3. **加载 tokenizer** — `Tokenizer.from_file(vocab_path, merges_path, special_tokens)`。
4. **分词** — 读取输入并对每个段落进行分词。
5. **写入输出** — 将 token ID 保存到输出文件。
6. **报告** — 将 token 计数和计时打印到 stderr。

## CLI 参数

| 参数 | 类型 | 必需 | 描述 |
|---|---|---|---|
| `input_path` | `str`（位置参数） | 是 | 输入文本文件路径 |
| `--vocab` / `-v` | `str` | 是 | vocab.json 路径（十六进制编码） |
| `--merges` / `-m` | `str` | 是 | merges.txt 路径（十六进制编码） |
| `--output` / `-o` | `str` | 是 | 输出 token ID 文件路径 |
| `--special-tokens` | `str` (nargs="*") | 否 | 训练时使用的特殊 token |
| `--stream` | flag | 否 | 逐行流式输出（用于大文件） |

## 集成点

- **`cs336_basics.tokenizer.Tokenizer.from_file(vocab_filepath, merges_filepath, special_tokens)`**
  - 类方法，读取十六进制编码的词表和合并文件并返回 `Tokenizer` 实例。
  - 注意：当前的 `from_file` 实现期望纯文本格式（每行 token + id 用于词表，每行一个 pair 用于合并），**不是** `save_tokenizer` 生成的十六进制编码 JSON 格式。你可能需要：
    - 编写一个单独的加载器来读取 `vocab.json`（带十六进制键的 JSON → `int: bytes`）和 `merges.txt`（十六进制 pair）。
    - 或者调整脚本以使用与 `train_bpe` 相同的十六进制格式。
- **`tokenizer.encode(text: str) -> list[int]`**
  - 将单个字符串编码为 token ID。
- **`tokenizer.encode_iterable(iterable: Iterable[str]) -> Iterable[int]`**
  - 用于可迭代对象的流式编码器 — 适用于逐行处理。

## 输入/输出格式

### 输入
- 纯 UTF-8 文本文件。
- 每行通常是一个文档或段落。

### 输出
- 每个输入段一行。
- 每行包含空格分隔的整数 token ID。
- 示例：
  ```
  245 18 9034 2 671 89
  18 4451 2
  ```

## 边界情况与错误处理

- **空输入文件** → 生成空的输出文件（零行）。
- **输入中的空行** → 跳过或在输出中生成空行（保持文档一致性）。
- **输入中的未知字节/token** — BPE 算法可以编码任何字节序列，因此不应发生；如果 `tokenizer.encode` 抛出异常，捕获它并报告有问题的行。
- **词表/合并文件未找到** → 捕获 `FileNotFoundError`，打印描述性消息，退出代码 1。
- **词表/合并文件格式错误** → 捕获 `json.JSONDecodeError` 或 `ValueError`，报告文件和行号，退出代码 1。
- **非常长的行** → `encode` 方法一次处理一个字符串；对于数 GB 的单个行，内存可能成为问题。对大输入使用 `--stream` 模式和 `encode_iterable`。
- **输入中的特殊 token** — tokenizer 已处理它们；确保通过 `--special-tokens` 传递它们以便原子性地保留。

## 行业标准

- 遵循 Unix 过滤器惯例：从文件（或 stdin 使用 `-`）读取，写入文件（或 stdout）。
- 将进度信息打印到 stderr，分词输出打印到 stdout（或指定的输出文件）。
- 在处理大文件时使用 `tqdm` 显示进度。
- 可选增强：透明地支持 gzip 压缩输入。
- 完成后报告总 token 数和处理速度（tokens/second）。
- 优雅地处理 `KeyboardInterrupt`（刷新部分输出，干净退出）。

## 流式处理考量

对于无法放入内存的大型语料库：

1. 打开输出文件和输入文件。
2. 使用 `for line in input_file:` 遍历输入行。
3. 每行调用 `tokenizer.encode(line.rstrip("\n"))`。
4. 将 `" ".join(map(str, token_ids)) + "\n"` 写入输出。
5. `tokenizer.encode_iterable` 方法专为此模式设计 — 使用它。

## 测试关注点

- 使用小型已知语料库测试：分词，然后解码，并验证往返过程保留了原始文本（除了空白规范化）。
- 验证输入中的特殊 token 作为原子 token ID 被保留。
- 检查输出文件格式是否符合预期（每个输入段一行，空格分隔的整数）。
- 在约 100 行的文件上测试 `--stream` 和默认模式，并确认输出相同。
