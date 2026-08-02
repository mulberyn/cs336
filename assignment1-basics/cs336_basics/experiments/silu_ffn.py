import torch
from torch import nn

from cs336_basics.modules import (
    RoPE, MultiHeadSelfAttention, SiLU, Embedding, Linear, RMSNorm
)
from cs336_basics.trainer import *

DEFAULT_CONFIG = {
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
    'save_ckp_path': './checkpoints/silu_ffn',
    'resume_ckp': None,
    'seed': 42, 
    
    # 数据配置
    'train_data_path': './data/TinyStoriesV2-GPT4-train.bin',
    'valid_data_path': './data/TinyStoriesV2-GPT4-valid.bin',
    'device': 'auto',
    
    # 记录配置
    'wandb_project': 'cs336-transformer',
    'wandb_run_name': 'experiment4.silu_ffn',
    'no_wandb': False,
    'export_final': './out/model/model_silu_ffn.pt',   # 最终模型导出路径
}


class TransformerBlockSiLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int, 
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        d_k = d_model // num_heads
        rope = RoPE(theta, d_k, max_seq_len, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoding=rope,
            device=device,
            dtype=dtype
        )
        self.ffn = SiLU(d_model, d_ff, device=device, dtype=dtype)
    

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        # Post Norm
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerLMSiLU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockSiLU(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.output_embedding = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.token_embedding(x)  # (batch, seq_len, d_model)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # (batch, seq_len, d_model)
        x = self.output_norm(x)  # (batch, seq_len, d_model)
        x = self.output_embedding(x)  # (batch, seq_len, vocab_size)
        return x


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

if __name__ == "__main__":
    set_seed(DEFAULT_CONFIG['seed'])
    device = get_device(DEFAULT_CONFIG['device'])
    train_data = np.memmap(DEFAULT_CONFIG['train_data_path'], dtype=np.uint32)
    val_data = np.memmap(DEFAULT_CONFIG['valid_data_path'], dtype=np.uint32)
    
    model = TransformerLMSiLU(
        vocab_size=DEFAULT_CONFIG['vocab_size'],
        context_length=DEFAULT_CONFIG['context_length'],
        d_model=DEFAULT_CONFIG['d_model'],
        num_layers=DEFAULT_CONFIG['num_layers'],
        num_heads=DEFAULT_CONFIG['num_heads'],
        d_ff=DEFAULT_CONFIG['d_ff'],
        rope_theta=DEFAULT_CONFIG['rope_theta'],
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=DEFAULT_CONFIG['lr_max'],
        betas=(DEFAULT_CONFIG['beta1'], DEFAULT_CONFIG['beta2']),
        eps=DEFAULT_CONFIG['eps'],
        weight_decay=DEFAULT_CONFIG['weight_decay'],
    )
    
    if DEFAULT_CONFIG['no_wandb']:
        wandb.init(mode='disabled')
    else:
        wandb.init(
            project=DEFAULT_CONFIG['wandb_project'],
            name=DEFAULT_CONFIG['wandb_run_name'],
            config=vars(DEFAULT_CONFIG),
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
        print(f"W&B 初始化成功: {wandb.run.name}")
    
    start_step = 0
    if DEFAULT_CONFIG['resume_ckp'] is not None:
        start_step = load_checkpoint(DEFAULT_CONFIG['resume_ckp'], model, optimizer)
        print(f"从 {DEFAULT_CONFIG['resume_ckp']} 恢复，从第 {start_step} 步继续")
        
    init_val_loss = evaluate(model, val_data, DEFAULT_CONFIG['batch_size'], DEFAULT_CONFIG['context_length'], device, DEFAULT_CONFIG['val_batch'])
    print(f"初始验证损失: {init_val_loss:.4f}")
    if not DEFAULT_CONFIG['no_wandb']:
        wandb.log({"valid/loss": init_val_loss, "valid/perplexity": math.exp(init_val_loss), "step": 0})
        
    model.train()
    pbar = tqdm(range(0, DEFAULT_CONFIG['train_steps']), 
                desc="Training", 
                initial=0, 
                total=DEFAULT_CONFIG['train_steps'])
    for step in pbar:
        # ========== 获取当前批次 ==========
        inputs, targets = data_loading(train_data, DEFAULT_CONFIG['batch_size'], DEFAULT_CONFIG['context_length'], device)
        inputs = inputs.long().to(device)
        targets = targets.long().to(device)
        
        # ========== 前向传播（输出 logits，计算 loss） ==========
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        
        # ========== 反向传播（梯度清零、梯度反向传播） ==========
        optimizer.zero_grad()
        loss.backward()
        
        # ========== 梯度裁剪（并获取 l2 范数） ==========
        grad_norm = gradient_clipping(model.parameters(), DEFAULT_CONFIG['max_l2_norm'])
        
        # ========== 更新学习率 ==========
        lr = get_lr_cosine_schedule(
            step + 1, DEFAULT_CONFIG['lr_max'], DEFAULT_CONFIG['lr_min'], DEFAULT_CONFIG['t_warm'], DEFAULT_CONFIG['t_end']
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # ========== 优化器更新 ==========
        optimizer.step()
            
        # ========== 日志记录（每 log_intervals 步） ==========
        if (step + 1) % DEFAULT_CONFIG['log_intervals'] == 0:
            perplexity = math.exp(loss.item())
            # 计算吞吐量（需要记录时间）
            # tokens_per_sec = (batch_size * context_length) / elapsed_time
            # 这里假设你已经计算了 elapsed_time
            print(f"Step {step + 1}/{DEFAULT_CONFIG['train_steps']} | loss={loss.item():.4f} | ppl={perplexity:.2f} | lr={lr:.2e} | grad_norm={grad_norm:.2f}")
            if not DEFAULT_CONFIG['no_wandb']:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/perplexity": perplexity,
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm,
                    "step": step + 1,
                })
        
        # ========== 验证（每 val_interval 步） ==========
        if (step + 1) % DEFAULT_CONFIG['val_interval'] == 0:
            val_loss = evaluate(model, val_data, DEFAULT_CONFIG['batch_size'], DEFAULT_CONFIG['context_length'], device, DEFAULT_CONFIG['val_batch'])
            val_ppl = math.exp(val_loss)
            print(f"Step {step+1} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            if not DEFAULT_CONFIG['no_wandb']:
                wandb.log({
                    "valid/loss": val_loss,
                    "valid/perplexity": val_ppl,
                    "step": step + 1,
                })
        
    final_loss = loss.item()   # 最后一次训练的 loss
    final_step = DEFAULT_CONFIG['train_steps']
    export_path = DEFAULT_CONFIG['export_final']
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    model.cpu()
    export_dict = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": DEFAULT_CONFIG['vocab_size'],
            "context_length": DEFAULT_CONFIG['context_length'],
            "d_model": DEFAULT_CONFIG['d_model'],
            "num_layers": DEFAULT_CONFIG['num_layers'],
            "num_heads": DEFAULT_CONFIG['num_heads'],
            "d_ff": DEFAULT_CONFIG['d_ff'],
            "rope_theta": DEFAULT_CONFIG['rope_theta'],
        },
        "final_loss": final_loss,
        "final_step": final_step,
    }
    torch.save(export_dict, export_path)
    print(f"最终模型已保存至 {export_path}")
    if not DEFAULT_CONFIG['no_wandb']:
        artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
        artifact.add_file(export_path)
        wandb.log_artifact(artifact)
        wandb.log({"final_loss": final_loss, "final_step": final_step})

    wandb.finish()