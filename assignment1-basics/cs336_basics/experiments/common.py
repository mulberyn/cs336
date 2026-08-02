import torch
from torch import nn
from typing import Literal

from cs336_basics.modules import (
    RoPE, MultiHeadSelfAttention, SwiGLU, SiLU, Embedding, Linear, RMSNorm
)
from cs336_basics.trainer import *

BASE_CONFIG = {
    # 模型配置
    'vocab_size': 10000,
    'context_length': 256,
    'd_model': 512,
    'num_layers': 4,
    'd_ff': 1344,
    'num_heads': 16,
    'rope_theta': 10000.0,
    
    # 优化器配置
    'lr_max': 1e-3,
    'lr_min': 1e-4,
    't_warm': 500,
    't_end': 10000,
    'weight_decay': 1e-2,
    'beta1': 0.9,
    'beta2': 0.95,
    'eps': 1e-8,
    'max_l2_norm': 1.0,
    
    # 训练配置
    'batch_size': 32,
    'train_steps': 6000,
    'val_interval': 100,
    'val_batch': 10,
    'save_intervals': 1000,
    'log_intervals': 1,
    'save_ckp_path': './checkpoints/', # need cover
    'resume_ckp': None,
    'seed': 42, 
    
    # 数据配置
    'train_data_path': './data/TinyStoriesV2-GPT4-train.bin',
    'valid_data_path': './data/TinyStoriesV2-GPT4-valid.bin',
    'device': 'auto',
    
    # 记录配置
    'wandb_project': 'cs336-transformer',
    'wandb_run_name': 'experiment', # need cover
    'no_wandb': False,
    'export_final': './out/model/experiment.pt',   # need cover
    
    "use_rmsnorm": True,
    "norm_position": "pre",
    "use_rope": True, 
    "ffn_type": "swiglu",
}

NormPosition = Literal["pre", "post"]
FFNType = Literal["swiglu", "silu"]

class AblationTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device,
        dtype: torch.dtype,
        use_rmsnorm: bool,
        norm_position: NormPosition,
        use_rope: bool,
        ffn_type: FFNType,
    ):
        super().__init__()
        self.norm_position = norm_position
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device, dtype=dtype) if use_rope else None
        
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoding=rope,
            device=device,
            dtype=dtype,
        )

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_position == "pre":
            out = x + self.attn(self.norm1(x))
            return out + self.ffn(self.norm2(out))
        if self.norm_position == "post":
            out = self.norm1(x + self.attn(x))
            return self.norm2(out + self.ffn(out))
        raise ValueError(f"Unknown norm_position: {self.norm_position}")


class AblationTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device,
        dtype: torch.dtype,
        use_rmsnorm: bool,
        norm_position: NormPosition,
        use_rope: bool,
        ffn_type: FFNType,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                AblationTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                    use_rmsnorm=use_rmsnorm,
                    norm_position=norm_position,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.out = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        return self.out(out)

import random
import wandb
import math
import os
from tqdm import tqdm
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg


def evaluate(model, dataset, batch_size, context_length, device, num_batches):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = data_loading(dataset, batch_size, context_length, device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def run(config) -> None:
    """
    根据 config 执行完整的训练流程。
    config 应包含所有必需的键（参见 BASE_CONFIG）。
    """
    # 1. 设置种子和设备
    set_seed(config["seed"])
    device_str = get_device(config["device"])
    device = torch.device(device_str)
    print(f"使用设备: {device}")

    # 2. 加载数据（内存映射）
    train_path = config["train_data_path"]
    valid_path = config["valid_data_path"]
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        raise FileNotFoundError("训练或验证数据文件不存在，请检查路径。")
    train_data = np.memmap(train_path, dtype=np.uint32, mode='r')
    val_data = np.memmap(valid_path, dtype=np.uint32, mode='r')
    print(f"训练数据大小: {len(train_data)} tokens")
    print(f"验证数据大小: {len(val_data)} tokens")

    # 3. 构建模型
    model = AblationTransformer(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
        dtype=torch.float32,
        use_rmsnorm=config["use_rmsnorm"],
        norm_position=config["norm_position"],
        use_rope=config["use_rope"],
        ffn_type=config["ffn_type"],
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 4. 构建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr_max"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )

    # 5. 恢复 checkpoint（若指定）
    start_step = 0
    if config.get("resume_ckp") is not None and os.path.exists(config["resume_ckp"]):
        start_step = load_checkpoint(config["resume_ckp"], model, optimizer)
        print(f"从 {config['resume_ckp']} 恢复，从第 {start_step} 步继续")

    # 6. 初始化 WandB
    if config["no_wandb"]:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            config=config,
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
        # 记录参数量
        wandb.config.update({"model/total_parameters": total_params}, allow_val_change=True)
        print(f"W&B 运行名称: {wandb.run.name}")

    # 7. 初始验证
    init_val_loss = evaluate(
        model, val_data, config["batch_size"],
        config["context_length"], device, config["val_batch"]
    )
    print(f"初始验证损失: {init_val_loss:.4f}")
    if not config["no_wandb"]:
        wandb.log({"valid/loss": init_val_loss, "valid/perplexity": math.exp(init_val_loss), "step": 0})

    # 8. 创建保存目录
    os.makedirs(config["save_ckp_path"], exist_ok=True)
    export_dir = os.path.dirname(config["export_final"])
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    # 9. 训练循环
    model.train()
    pbar = tqdm(
        range(start_step, config["train_steps"]),
        desc="Training",
        initial=start_step,
        total=config["train_steps"],
    )
    for step in pbar:
        # 获取批次
        inputs, targets = data_loading(
            train_data, config["batch_size"],
            config["context_length"], device
        )
        # 确保类型为 long
        inputs = inputs.long()
        targets = targets.long()

        # 前向与损失
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪并获取范数
        grad_norm = gradient_clipping(model.parameters(), config["max_l2_norm"])

        # 学习率调度
        lr = get_lr_cosine_schedule(
            step + 1,
            config["lr_max"],
            config["lr_min"],
            config["t_warm"],
            config["t_end"],
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 优化器更新
        optimizer.step()

        # 日志记录
        if (step + 1) % config["log_intervals"] == 0:
            perplexity = math.exp(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{perplexity:.2f}",
                "lr": f"{lr:.2e}",
                "grad": f"{grad_norm:.2f}",
            })
            if not config["no_wandb"]:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/perplexity": perplexity,
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm,
                    "step": (step + 1),
                })

        # 验证
        if (step + 1) % config["val_interval"] == 0:
            val_loss = evaluate(
                model, val_data, config["batch_size"],
                config["context_length"], device, config["val_batch"]
            )
            val_ppl = math.exp(val_loss)
            print(f"\nStep {step + 1} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            if not config["no_wandb"]:
                wandb.log({
                    "valid/loss": val_loss,
                    "valid/perplexity": val_ppl,
                    "step": step + 1,
                })

        # 保存 checkpoint
        if (step + 1) % config["save_intervals"] == 0:
            ckpt_path = os.path.join(
                config["save_ckp_path"],
                f"checkpoint_{step + 1:08d}.pt"
            )
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            print(f"Checkpoint 保存至: {ckpt_path}")

    # 10. 训练结束，保存最终模型
    final_loss = loss.item()
    final_step = config["train_steps"]
    export_path = config["export_final"]

    # 移至 CPU 再保存，方便跨设备加载
    model.cpu()
    export_dict = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": config["vocab_size"],
            "context_length": config["context_length"],
            "d_model": config["d_model"],
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "d_ff": config["d_ff"],
            "rope_theta": config["rope_theta"],
            "use_rmsnorm": config["use_rmsnorm"],
            "norm_position": config["norm_position"],
            "use_rope": config["use_rope"],
            "ffn_type": config["ffn_type"],
        },
        "final_loss": final_loss,
        "final_step": final_step,
    }
    torch.save(export_dict, export_path)
    print(f"最终模型已保存至: {export_path}")

    # 11. 上传至 WandB（可选）
    if not config["no_wandb"]:
        artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
        artifact.add_file(export_path)
        wandb.log_artifact(artifact)
        wandb.log({"final_loss": final_loss, "final_step": final_step})

    wandb.finish()