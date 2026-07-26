# Tokenizer 调试报告

**日期：** 2026-07-23
**背景：** 针对 `tests/test_tokenizer.py` 测试套件调试 `cs336_basics/tokenizer/tokenizer.py`。
**初始测试结果：** 21 个失败，2 个通过，2 个跳过

---

## 问题根因总结

全部 21 个测试失败源于 `tokenizer.py` 中的 **三个相互关联的问题**：

| # | 问题 | 位置 | 严重程度 |
|---|------|------|----------|
| 1 | vocab 数据结构方向不匹配 | `__init__` + `encode` | 严重 — 导致所有 `KeyError` 失败 |
| 2 | 缺少 `@classmethod` 装饰器 | `from_file` | 中等 — 破坏类方法调用约定 |
| 3 | 类型注解错误 | `__init__` 签名 | 较低 — 注解与实际数据流不一致 |

---

## 问题一：vocab 数据结构方向不匹配

### 测试代码实际传入的数据

测试适配器（`tests/adapters.py`，第 544–564 行）使用以下参数构造 tokenizer：

- **`vocab`**：`dict[int, bytes]` — token ID（int）→ token 字节序列
  - 示例：`{0: b'!', 1: b'"', ..., 10: b'+', ...}`
  - 全部 256 个单字节 token 均作为 value 存在
  - 特殊 token（如 `<|endoftext|>`）作为额外条目追加
- **`merges`**：`list[tuple[bytes, bytes]]` — 有序的字节对合并列表
  - 示例：`[(b' ', b't'), (b' ', b'a'), (b'h', b'e'), ...]`

### 代码中对 `vocab` 的使用方式

`Tokenizer` 类对 `self.vocab` 的使用存在 **两个相反方向**：

| 方法 | 查找操作 | 期望方向 | 与 `dict[int, bytes]` 兼容？ |
|------|----------|----------|-------------------------------|
| `encode`（第 68 行） | `self.vocab[token.encode("utf-8")]` | bytes → int | **否** — key 是 `bytes`，字典 key 是 `int` |
| `encode`（第 91 行） | `self.vocab[tok]` | bytes → int | **否** — key 是 `bytes`，字典 key 是 `int` |
| `decode`（第 116 行） | `self.vocab[id]` | int → bytes | **是** — key 是 `int`，与字典匹配 |

### 为什么每个测试都因 `KeyError` 失败

以 `test_roundtrip_single_character` 为例，编码 `"s"` 的过程：

1. `pretokenize("s")` 返回 `["s"]`
2. 字符串 `"s"` 编码为 UTF-8 → `b's'`
3. 每个字节变为独立 token：`[b's']`
4. 无需合并（单字节），因此循环直接退出
5. 第 91 行：`self.vocab[b's']` — 但 `self.vocab` 的 key 是 int，不是 bytes → **`KeyError`**

同样的模式导致所有其他失败。字符各不相同（`b'\n'`、`b'H'`、`b'Die'` 等），但根因完全一致：**在以 int 为 key 的字典中查找 bytes 类型的 key**。

### 修复思路

`encode` 方法需要完成 token 字节序列 → token ID 的查找。由于词汇表存储为 ID → 字节序列，你需要在 `__init__` 中构建一个 **反向映射**。例如，添加 `self.vocab_reverse: dict[bytes, int]`，将每个字节序列映射回其 ID。

关键注意事项：
- 反向映射应在 `__init__` 中构建一次，而非每次调用 `encode` 时重复构建
- 特殊 token 查找（第 68 行）和普通 token 查找（第 91 行）都应使用反向映射
- `decode` 配合 ID → 字节序列方向已经能正常工作 — **不要修改它**

---

## 问题二：`from_file` 缺少 `@classmethod` 装饰器

### 问题所在

第 18 行，`from_file` 的定义如下：

```python
def from_file(
    cls,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str] | None = None
):
```

第一个参数名为 `cls`，但没有 `@classmethod` 装饰器。没有它：
- 调用 `Tokenizer.from_file(...)` 时参数会按位置传递 — 第一个参数（`vocab_filepath`）会绑定到 `cls`，而 `merges_filepath` 会变成 `vocab_filepath`，造成令人困惑的错误
- 该方法被设计为替代构造函数（其返回 `cls(...)`），但无法正确工作

### 修复思路

在 `def from_file` 的前一行加上 `@classmethod` 装饰器。

---

## 问题三：类型注解错误

### `vocab` 注解（第 8 行）

当前：`vocab: dict[str, int]`
应为：`vocab: dict[int, bytes]`

测试始终传入 `{int: bytes}`，这与 `train.py` 中的约定一致（参见 `initialize_vocab`，返回 `dict[int, bytes]`；以及 `train_bpe`，返回 `tuple[dict[int, bytes], ...]`）。

### `merges` 注解（第 9 行）

当前：`merges: list[tuple[str, str]]`
应为：`merges: list[tuple[bytes, bytes]]`

merges 文件格式使用 GPT-2 的字节到 Unicode 重映射。测试适配器将其解码回原始字节后才传入。`pair_to_priority` 字典和合并循环都已在正确处理 `tuple[bytes, bytes]` key — 只是注解写错了。

---

## 逐项测试失败分析

全部 21 个失败共享同一个 `KeyError` 根因（问题一）。分类如下：

### 单字符编码（4 个测试）
- `test_roundtrip_single_character` → `KeyError: b's'`
- `test_single_character_matches_tiktoken` → `KeyError: b's'`
- `test_roundtrip_single_unicode_character` → `KeyError: b'...'`
- `test_single_unicode_character_matches_tiktoken` → `KeyError: b'...'`

ASCII 字节必须在反向 vocab 中查找。Unicode 字符会产生多字节的 UTF-8 序列；每个字节 token 都需要反向查找。

### 字符串编码（10 个测试）
- `test_roundtrip_ascii_string` → `KeyError: b'Hello'`（第一个多字节 token）
- `test_ascii_string_matches_tiktoken` → 同上
- `test_roundtrip_unicode_string` → `KeyError: b'H'`
- `test_unicode_string_matches_tiktoken` → 同上
- `test_roundtrip_unicode_string_with_special_tokens` → 同上
- `test_unicode_string_with_special_tokens_matches_tiktoken` → 同上
- `test_overlapping_special_tokens` → 同上

全部在 `encode` 中首次 token 查找时失败。

### 文件语料测试（6 个测试）
- `test_address_roundtrip` → `KeyError: b'Four'`
- `test_address_matches_tiktoken` → 同上
- `test_german_roundtrip` → `KeyError: b'Die'`
- `test_german_matches_tiktoken` → 同上
- `test_tinystories_sample_roundtrip` → 同样模式
- `test_tinystories_matches_tiktoken` → 同样模式

### 特殊 token 边界情况测试（2 个测试）
- `test_encode_special_token_trailing_newlines` → 同样根因
- `test_encode_special_token_double_newline_non_whitespace` → 同样根因

### 迭代器测试（2 个测试）
- `test_encode_iterable_tinystories_sample_roundtrip` → 同样根因，经由 `encode_iterable`
- `test_encode_iterable_tinystories_matches_tiktoken` → 同样根因，经由 `encode_iterable`

### 通过的测试（2 个）
- `test_roundtrip_empty` — 通过，因为 `encode("")` 在第 60-61 行的提前返回语句中直接返回 `[]`，从未到达有问题的字典查找
- `test_empty_matches_tiktoken` — 同样原因

---

## 修复验证清单

修改代码后，请验证以下内容：

1. 全部 23 个测试通过（21 个当前失败 + 2 个当前通过）
2. `encode` 能正确查找单字节和多字节合并 token
3. `decode` 仍能正确将 ID 映射回字节序列
4. 特殊 token（如 `<|endoftext|>`）能正确编码与解码
5. 来回一致性成立：`decode(encode(text)) == text`
6. 输出与 tiktoken 的 GPT-2 编码在相同输入下匹配

---

## 数据流全景

以下是 tokenizer 中完整数据流的参考图：

```
测试适配器 (adapters.py)
  │
  │  构造:
  │    vocab:  {int_id → bytes}
  │    merges: [(bytes, bytes), ...]
  │
  ▼
Tokenizer.__init__(vocab, merges, special_tokens)
  │
  │  应同时存储:
  │    self.vocab:             int → bytes   (供 decode 使用)
  │    self.reverse_vocab:     bytes → int   (供 encode 使用，需自行构建)
  │    self.pair_to_priority:  (bytes, bytes) → int  (控制合并优先级)
  │
  ▼
encode(text)
  │
  │  1. pretokenize(text, special_tokens) → 预分割 token 字符串列表
  │  2. 对每个预分割 token:
  │     a. 若是特殊 token → 在 reverse_vocab 中查找
  │     b. 否则:
  │        - 拆分为单字节 token
  │        - 按优先级顺序应用合并（使用 pair_to_priority）
  │        - 在 reverse_vocab 中查找每个最终 token
  │
  ▼
decode(ids)
  │
  │  对每个 ID，查找 self.vocab[id] → bytes
  │  拼接全部字节，以 UTF-8 解码
  │
  ▼
结果: string
```

**核心要点：** `encode` 需要 bytes → int 映射，`decode` 需要 int → bytes 映射。测试提供的是 int → bytes。你的 `__init__` 需要原样存储 int → bytes（供 `decode` 使用），同时构建反向的 bytes → int 映射（供 `encode` 使用）。
