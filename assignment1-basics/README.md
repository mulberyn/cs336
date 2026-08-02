# CS336 2026 年春季学期 assignment1-basics

有关作业的完整说明，请参阅作业说明文档
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

## 环境设置

### 环境

我们使用 `uv` 管理项目环境，以确保可复现性、可移植性和易用性。
安装 `uv`（[官方安装指南](https://github.com/astral-sh/uv#installation)，推荐），或运行 `pip install uv` / `brew install uv`。
建议先阅读一些关于用 `uv` 管理项目的[文档](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)（你不会后悔的！）。

现在你可以使用以下命令运行仓库中的任意代码：

```sh
uv run <python_file_path>
```

必要时会自动解析并激活环境。

### 运行单元测试

```sh
uv run pytest
```

初始状态下，所有测试都应因 `NotImplementedError` 而失败。
要将你的实现与测试连接起来，请补全
[./tests/adapters.py](./tests/adapters.py) 中的函数。

### 下载数据

下载 TinyStories 数据集以及 OpenWebText 的一个子集：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## 训练与使用流程

以下命令全部在 **`assignment1-basics/`** 目录下执行。脚本中的 `./data/...`、`./out/...` 都是相对该目录的路径，使用 `uv run` 会自动激活项目环境。

完整流程共 4 步，环环相扣：

1. **训练 BPE tokenizer**（`train_bpe.py`）→ 得到词表与合并规则
2. **把原始文本分词为 `.bin` 数据**（`tokenize_text.py`）→ 得到训练/验证数据
3. **训练 Transformer 模型**（`train_model.py`）→ 得到最终模型
4. **用训练好的模型生成文本**（`generate.py`）

### 第 1 步：训练 BPE tokenizer

```sh
uv run python cs336_basics/scripts/train_bpe.py
```

默认参数：

| 参数               | 默认值                                | 说明                  |
| ------------------ | ------------------------------------- | --------------------- |
| `--input_path`     | `./data/TinyStoriesV2-GPT4-train.txt` | UTF-8 训练语料路径    |
| `--vocab_size`     | `10000`                               | 目标词表大小（≥ 256） |
| `--special_tokens` | `<\|endoftext\|>`                     | 特殊 token 列表       |
| `--tokenizer_dir`  | `./out/tokenizer`                     | tokenizer 保存目录    |
| `--num_workers`    | `1`                                   | 并行预分词 worker 数  |

**产物**：`./out/tokenizer/vocab.json`、`./out/tokenizer/merges.json`

### 第 2 步：把文本分词为 `.bin` 数据

默认命令（处理训练集）：

```sh
uv run python cs336_basics/scripts/tokenize_text.py
```

验证集需要显式指定输入/输出路径（tokenizer 用第 1 步默认的目录即可）：

```sh
uv run python cs336_basics/scripts/tokenize_text.py \
    --input_path ./data/TinyStoriesV2-GPT4-valid.txt \
    --output_path ./data/TinyStoriesV2-GPT4-valid.bin
```

参数说明：

| 参数              | 默认值                                | 说明                                      |
| ----------------- | ------------------------------------- | ----------------------------------------- |
| `--input_path`    | `./data/TinyStoriesV2-GPT4-valid.txt` | 待分词的原始文本                          |
| `--output_path`   | `./data/TinyStoriesV2-GPT4-valid.bin` | 输出的 token ID 序列文件（uint32 二进制） |
| `--tokenizer_dir` | `./out/tokenizer`                     | 加载的 tokenizer 目录                     |

**产物**：`./data/TinyStoriesV2-GPT4-train.bin`、`./data/TinyStoriesV2-GPT4-valid.bin`

### 第 3 步：训练模型

```sh
uv run python cs336_basics/scripts/train_model.py
```

默认参数（按配置分组）：

| 参数                                      | 默认值                                               | 说明                            |
| ----------------------------------------- | ---------------------------------------------------- | ------------------------------- |
| `--vocab_size`                            | `10000`                                              | 词表大小（需与 tokenizer 一致） |
| `--context_length`                        | `256`                                                | 最大序列长度                    |
| `--d_model`                               | `512`                                                | 模型维度                        |
| `--num_layers`                            | `4`                                                  | Transformer 层数                |
| `--num_heads`                             | `16`                                                 | 注意力头数                      |
| `--d_ff`                                  | `1344`                                               | FFN 维度                        |
| `--lr_max` / `--lr_min`                   | `1e-3` / `1e-4`                                      | 最大/最小学习率                 |
| `--t_warm` / `--t_end`                    | `500` / `10000`                                      | 预热步数 / 余弦退火总步数       |
| `--weight_decay`                          | `1e-2`                                               | 权重衰减                        |
| `--beta1` / `--beta2` / `--eps`           | `0.9` / `0.95` / `1e-8`                              | Adam 超参数                     |
| `--max_l2_norm`                           | `1.0`                                                | 梯度裁剪阈值                    |
| `--batch_size`                            | `32`                                                 | 批大小                          |
| `--train_steps`                           | `6000`                                               | 总训练步数                      |
| `--val_interval`                          | `100`                                                | 每多少步验证一次                |
| `--save_intervals`                        | `1000`                                               | 每多少步保存一次 checkpoint     |
| `--resume_ckp`                            | `None`                                               | 从 checkpoint 恢复训练          |
| `--train_data_path` / `--valid_data_path` | `./data/TinyStoriesV2-GPT4-train.bin` / `-valid.bin` | 第 2 步生成的训练/验证数据      |
| `--device`                                | `auto`                                               | `auto` / `cpu` / `cuda` / `mps` |
| `--no_wandb`                              | 关闭                                                 | 加上该选项则禁用 W&B 记录       |
| `--export_final`                          | `./out/model/model_final.pt`                         | 最终模型导出路径                |

**产物**：

- checkpoint：`./checkpoints/checkpoint_<step>.pt`（可用于 `--resume_ckp` 恢复）
- 最终模型：`./out/model/model_final.pt`（第 4 步默认读取该路径）

### 第 4 步：用训练好的模型生成文本

```sh
uv run python cs336_basics/scripts/generate.py
```

参数说明：

| 参数               | 默认值                       | 说明                            |
| ------------------ | ---------------------------- | ------------------------------- |
| `--model_path`     | `./out/model/model_final.pt` | 第 3 步导出的模型文件           |
| `--tokenizer_dir`  | `./out/tokenizer`            | tokenizer 目录                  |
| `--prompt`         | `Once upon a time,`          | 生成的前缀文本                  |
| `--max_new_tokens` | `200`                        | 最大生成 token 数               |
| `--temperature`    | `0.8`                        | 采样温度（`0` 表示贪婪采样）    |
| `--top_p`          | `0.9`                        | Top-p 截断（`1.0` 表示不使用）  |
| `--stop_token`     | `<\|endoftext\|>`            | 遇到该 token 停止生成           |
| `--device`         | `auto`                       | `auto` / `cpu` / `cuda` / `mps` |
