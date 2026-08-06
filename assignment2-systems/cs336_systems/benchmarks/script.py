import torch
import torch.nn.functional as F
import timeit
import sys
import numpy as np
from typing import TypedDict, Literal, Optional

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


DEFAULT_CONFIG = {
    'vocab_size': 10000,
    'context_length': 256,
    'd_model': 512,
    'num_layers': 4,
    'd_ff': 1344,
    'num_heads': 16,
    'rope_theta': 10000.0,
    'batch_size': 4,
    'max_learning_rate': 1e-3,
    'min_learning_rate': 1e-4,
    'warmup_iters': 500,
    'cosine_cycle_iters': 10000,
    'weight_decay': 1e-2,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'it_warm': 5,                        # 预热步数
    'it_n': 10,                          # 测量步数
    'mode': 'full',                      # 'forward', 'forward_backward', 'full'
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'verbose': True,
}


def create_batch(configs: dict, device: str):
    """生成一个随机 batch 的输入和目标，用于模型训练/推理"""
    batch_size = configs['batch_size']
    seq_len = configs['context_length']
    vocab_size = configs['vocab_size']
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return inputs, targets


def benchmark_step(model, optimizer, inputs, targets, mode: str, device: str):
    """
    执行一步训练或推理，并根据 mode 决定执行哪些操作。
    mode:
        'forward' : 只做前向传播，不计损失和反向
        'forward_backward' : 前向 + 损失 + 反向
        'full' : 前向 + 损失 + 反向 + 优化器更新
    """
    optimizer.zero_grad(set_to_none=True)
    
    # 前向传播
    logits = model(inputs)                     # shape: (batch, seq_len, vocab_size)
    
    if mode == 'forward':
        return
    
    loss = cross_entropy(logits, targets)
    
    if mode == 'forward_backward':
        loss.backward()
        return
    
    if mode == 'full':
        loss.backward()
        optimizer.step()
        return


def main(configs: dict) -> int:
    device = configs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    it_warm = configs['it_warm']
    it_n = configs['it_n']
    mode = configs.get('mode', 'full')
    verbose = configs.get('verbose', True)
    
    model = BasicsTransformerLM(
        vocab_size=configs['vocab_size'],
        context_length=configs['context_length'],
        d_model=configs['d_model'],
        num_layers=configs['num_layers'],
        num_heads=configs['num_heads'],
        d_ff=configs['d_ff'],
        rope_theta=configs['rope_theta'],
    )
    model.to(device)
    
    optimizer = AdamW(
        params=model.parameters(),
        lr=configs['max_learning_rate'],
        betas=configs['betas'],
        eps=configs['eps'],
        weight_decay=configs['weight_decay'],
    )
    
    inputs, targets = create_batch(configs, device)
    
    model.train()
    for _ in range(it_warm):
        benchmark_step(model, optimizer, inputs, targets, mode, device)
        torch.cuda.synchronize() if device == 'cuda' else None
    
    timings = []
    timer = timeit.default_timer
    print(f"\nMeasuring {it_n} steps...")
    for step_idx in range(it_n):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = timer()
        benchmark_step(model, optimizer, inputs, targets, mode, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = timer()
        elapsed = end - start
        timings.append(elapsed)
        
        if verbose:
            print(f"  Step {step_idx+1:3d}/{it_n}: {elapsed*1000:8.3f} ms")
    
    timings_ms = np.array(timings) * 1000
    
    mean = np.mean(timings_ms)
    std = np.std(timings_ms, ddof=1)
    min_val = np.min(timings_ms)
    max_val = np.max(timings_ms)
    median = np.median(timings_ms)
    p95 = np.percentile(timings_ms, 95)
    p99 = np.percentile(timings_ms, 99)
    
    # 打印汇总
    print("\n" + "="*50)
    print(f"Mode: {mode}")
    print(f"Warmup steps: {it_warm}, Measurement steps: {it_n}")
    print(f"Average time per step: {mean:.3f} ms")
    print(f"Std deviation: {std:.3f} ms")
    print(f"Min: {min_val:.3f} ms, Max: {max_val:.3f} ms")
    print(f"Median: {median:.3f} ms")
    print(f"95th percentile: {p95:.3f} ms")
    print(f"99th percentile: {p99:.3f} ms")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    config = DEFAULT_CONFIG.copy()
    sys.exit(main(config))