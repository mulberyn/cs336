# 实现指南：`train_model.py`

## 概述

一个 CLI 脚本，在预分词数据上训练 `TransformerLM` 模型，支持完整的超参数配置、Weights & Biases 日志记录、checkpoint 保存、以及用于文本生成的模型导出。

这是作业 1 的核心脚本 — 它将所有模块（embedding、TransformerBlock、MultiHeadSelfAttention、SwiGLU、RMSNorm、RoPE、AdamW、余弦学习率调度、梯度裁剪、数据加载、checkpoint 保存）连接成一个完整的训练流水线。

## 职责

- 从磁盘加载预分词的训练数据（token ID 的 numpy 数组）。
- 使用指定的架构构建 `TransformerLM` 模型。
- 设置 AdamW 优化器、余弦学习率调度（含线性预热）和梯度裁剪。
- 运行训练循环：前向传播 → 损失计算 → 反向传播 → 梯度裁剪 → 优化器更新 → 学习率调度更新。
- 将训练指标（损失、困惑度、学习率、梯度范数、tokens/秒）记录到：
  - **控制台**（stderr）— 人类可读的进度。
  - **W&B**（wandb）— 带图表的结构化实验跟踪。
- 定期保存模型 checkpoint（完整的优化器状态和仅推理的导出）。
- 支持从之前的 checkpoint 恢复训练。
- 训练完成后，保存最终模型供后续解码/生成使用。

## 推荐结构

```
main()
├── parse_args()              # 所有超参数作为 CLI 标志
├── setup_wandb()             # 初始化 W&B 运行（配置、项目名）
├── set_seed()                # 可重复性
├── load_data()               # 加载预分词的 numpy 数组
├── build_model()             # 根据解析的配置构建 TransformerLM
├── build_optimizer()         # AdamW(params, lr, betas, eps, weight_decay)
├── build_scheduler()         # get_lr_cosine_schedule(...)
├── train_loop()              # 主训练循环
│   ├── for step in range(start_step, total_steps):
│   │   ├── get_batch()      # data_loading(dataset, batch_size, context_length, device)
│   │   ├── forward_pass()   # logits = model(inputs); loss = cross_entropy(logits, targets)
│   │   ├── backward_pass()  # loss.backward()
│   │   ├── gradient_clipping(parameters, max_norm)
│   │   ├── optimizer.step()
│   │   ├── scheduler.step() #（如果学习率每步变化）
│   │   ├── log_metrics()    # loss, lr, grad_norm, tokens_per_sec, perplexity
│   │   ├── wandb.log(...)
│   │   └── if step % save_interval == 0: save_checkpoint(...)
│   └── save_final_model()   # 导出用于解码
└── finalize_wandb()
```

## CLI 参数

所有超参数应通过 `argparse` 暴露。按类别分组以便清晰。

### 模型架构

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `--d-model` | `int` | 512 | 隐藏层维度 |
| `--num-layers` | `int` | 4 | Transformer 块的数量 |
| `--num-heads` | `int` | 8 | 注意力头数量 |
| `--d-ff` | `int` | — | FFN 隐藏层维度（默认：8/3 * d_model，对齐到 64） |
| `--rope-theta` | `float` | 10000.0 | RoPE 基频 |

### 训练超参数

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `--lr` | `float` | 3e-4 | 峰值学习率 |
| `--lr-min` | `float` | 1e-5 | 最小学习率（余弦退火目标） |
| `--warmup-steps` | `int` | 1000 | 线性预热步数 |
| `--total-steps` | `int` | — | 总训练步数 |
| `--batch-size` | `int` | 64 | 每批序列数 |
| `--context-length` | `int` | 256 | 序列长度（必须与数据预处理一致） |
| `--beta1` | `float` | 0.9 | Adam beta1 |
| `--beta2` | `float` | 0.95 | Adam beta2 |
| `--eps` | `float` | 1e-8 | Adam epsilon |
| `--weight-decay` | `float` | 0.1 | AdamW 权重衰减 |
| `--max-grad-norm` | `float` | 1.0 | 梯度裁剪阈值 |
| `--seed` | `int` | 42 | 随机种子 |

### 数据与 Checkpoint

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `--data-path` | `str` | — | 预分词 .npy 训练数据路径 |
| `--val-data-path` | `str` | — | 可选的验证 .npy（用于评估困惑度） |
| `--vocab-size` | `int` | — | 词表大小（必须与 tokenizer 匹配） |
| `--save-dir` | `str` | `./checkpoints` | checkpoint 和最终模型目录 |
| `--save-interval` | `int` | 1000 | checkpoint 保存间隔（步数） |
| `--resume` | `str` | — | 从中恢复的 checkpoint 路径 |
| `--log-interval` | `int` | 10 | 控制台/wandb 日志记录间隔（步数） |

### W&B

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `--wandb-project` | `str` | `"cs336-basics"` | W&B 项目名称 |
| `--wandb-run` | `str` | — | 可选的运行名称（为空则自动生成） |
| `--wandb-entity` | `str` | — | W&B 用户名/团队 |
| `--wandb-disable` | flag | — | 完全禁用 W&B 日志记录 |

### 推理导出

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `--export-final` | `str` | `./model_final.pt` | 最终模型保存路径（仅 state_dict，不含优化器） |

## 集成点

- **`cs336_basics.modules.TransformerLM`** — 完整模型类。
  - `TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device, dtype)`
- **`cs336_basics.trainer.AdamW`** — 自定义 AdamW 优化器。
  - `AdamW(params, lr=..., betas=(beta1, beta2), eps=..., weight_decay=...)`
- **`cs336_basics.trainer.cross_entropy`** — 损失函数。
  - `cross_entropy(logits, targets) -> scalar tensor`
- **`cs336_basics.trainer.get_lr_cosine_schedule`** — 学习率调度。
  - `get_lr_cosine_schedule(t, lr_max, lr_min, t_warm, t_end) -> float`
  - 每步调用此函数计算当前学习率，然后更新优化器的 `param_groups[0]['lr']`。
- **`cs336_basics.trainer.gradient_clipping`** — 梯度缩放。
  - `gradient_clipping(model.parameters(), max_l2_norm)`
- **`cs336_basics.trainer.data_loading`** — 批次采样。
  - `data_loading(dataset, batch_size, context_length, device) -> (inputs, targets)`
- **`cs336_basics.trainer.save_checkpoint` / `load_checkpoint`** — 持久化。
  - `save_checkpoint(model, optimizer, iteration, path)`
  - `iteration = load_checkpoint(path, model, optimizer)`

## 训练循环 — 关键设计决策

### 学习率调度
- `get_lr_cosine_schedule` 函数为给定步数 `t` 返回一个浮点数。
- 每一步，计算 `lr = get_lr_cosine_schedule(step, lr_max, lr_min, warmup_steps, total_steps)` 并赋值给 `optimizer.param_groups[0]['lr'] = lr`。
- 这避免了将优化器与外部调度器对象耦合。

### 损失与困惑度
- 损失是所有非填充 token 的平均交叉熵。
- 困惑度 = `exp(loss)` — 将其与损失一起记录以便解释。

### 梯度裁剪
- 在 `optimizer.step()` **之前**和 `loss.backward()` **之后**调用 `gradient_clipping(model.parameters(), max_grad_norm)`。

### 设备处理
- 将模型移动到适当的设备：`model.to(device)`。
- `data_loading` 函数已接受 `device` 参数 — 批次直接落在目标设备上。

### 混合精度（可选增强）
- 考虑在前向传播中添加 `torch.autocast(device_type, dtype=torch.float16)`。
- 这是标准做法，但对于作业是可选的。

## W&B 集成

### 初始化（训练循环之前）
```python
import wandb
wandb.init(
    project=args.wandb_project,
    name=args.wandb_run,
    entity=args.wandb_entity,
    config=vars(args),        # 记录所有超参数
)
```

### 逐步日志记录
```python
wandb.log({
    "loss": loss.item(),
    "perplexity": math.exp(loss.item()),
    "learning_rate": current_lr,
    "gradient_norm": grad_norm,
    "tokens_per_second": tokens_per_sec,
    "step": step,
})
```

- 每 `log_interval` 步记录一次，而不是每一步（避免压垮 W&B 服务器）。
- 记录梯度范数（在 `gradient_clipping` 内部或在裁剪之前单独计算）。
- 计算 tokens/sec 为 `(batch_size * context_length) / step_duration_seconds`。

### 结束
```python
wandb.finish()
```

### 配置
- 当设置了 `--wandb-disable` 时，调用 `wandb.init(mode="disabled")` 使所有 `wandb.log` 调用变为空操作，无需在训练循环中添加条件分支。
- 使用 `wandb.define_metric("step")` 和 `wandb.define_metric("*", step_metric="step")` 在图表中实现正确的 x 轴对齐。

## 模型导出用于解码

最终模型应保存在适合下游文本生成的格式中：

### 导出格式
```python
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": rope_theta,
    },
    "final_loss": loss.item(),
    "final_step": step,
}, args.export_final)
```

`model_config` 字典至关重要 — 它允许解码脚本重建模型架构而无需重新解析 CLI 标志。

## 边界情况与错误处理

- **从 checkpoint 恢复**：验证 checkpoint 的模型配置与当前 CLI 参数匹配（架构兼容性）。如果不匹配则打印警告。
- **数据太小**：如果 `len(dataset) < context_length + 1`，则无法采样有效的批次。在启动时检查并退出并给出清晰错误。
- **NaN 损失**：检测 `torch.isnan(loss)` 并记录警告。可选择在状态变得不可恢复之前触发 checkpoint 保存。
- **W&B 失败**：将 `wandb.init` 和 `wandb.log` 包装在 try/except 中，使网络故障不会导致训练崩溃。
- **磁盘空间**：在 checkpoint 保存期间捕获 `OSError` 并打印警告（但继续训练 — 下次保存可能成功）。
- **KeyboardInterrupt**：在 Ctrl+C 退出前保存最终 checkpoint。

## 可重复性

```python
def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

## 行业标准

- 使用带有参数组（`--model`、`--training`、`--data`、`--wandb` 等）的 `argparse.ArgumentParser` 以获得有组织的帮助输出。
- 将带有时间戳的日志输出到 stderr（例如 `[2026-07-30 14:30:01] Step 1000 | loss=3.45 lr=3e-4`）。
- 在训练循环中使用 `tqdm` 或结构化的日志行（而不是原始的 print 语句）。
- 在不中断训练的情况下保存中间 checkpoint（序列化到临时文件，然后原子性地重命名）。
- 保留最后 N 个 checkpoint 并删除较早的以管理磁盘使用。
- 将梯度范数作为健康信号记录 — 它应随时间稳定；突然的峰值通常先于发散。
- 报告 tokens per second — 这是 LLM 训练的关键吞吐量指标。

## 测试关注点

- 使用微小的模型（例如 `--d-model 64 --num-layers 2 --num-heads 4`）和小型数据集进行测试，验证训练循环收敛（损失在前几百步中单调递减）。
- 验证从 checkpoint 恢复产生与从头运行相同的损失曲线（即优化器状态被正确恢复）。
- 检查导出的模型文件可以被加载并产生合理的输出（不是 NaN）。
- 使用 `--wandb-disable` 运行以验证纯控制台模式正常工作。
- 通过人为创建大梯度并确认范数被裁剪来测试梯度裁剪。
