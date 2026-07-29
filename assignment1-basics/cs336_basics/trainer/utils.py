import torch
import math
from collections.abc import Iterable

def cross_entropy(
    out_logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    # 1. 数值稳定：减去最大值（注意：使用减法赋值给新变量，不修改原值）
    # dim=-1 表示对最后一个维度（词表维度）操作，兼容 2D 和 3D 输入
    max_val = out_logits.max(dim=-1, keepdim=True)[0]
    logits_shifted = out_logits - max_val  # 这里不是原地操作，是安全的
    
    # 2. 计算 log(sum(exp(logits_shifted)))
    # 此时最大值对应的 exp 为 1，sum 的范围在 [1, vocab_size]，绝对安全，不会溢出
    sum_exp = torch.exp(logits_shifted).sum(dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    
    # 3. 计算 log_softmax = logits - log(sum_exp)
    log_softmax = logits_shifted - log_sum_exp
    
    # 4. 提取出 targets 对应位置的对数概率
    # 使用 gather 或高级索引，这里保持你原来的写法
    log_probs = log_softmax[range(out_logits.shape[0]), targets]
    
    # 5. 返回平均交叉熵损失（注意：根据前一个报错，你应该用 mean，而不是 sum）
    return -torch.mean(log_probs)


def get_lr_cosine_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    t_warm: int,
    t_end: int
) -> float:
    if t < t_warm:
        return t / t_warm * lr_max
    elif t <= t_end:
        return lr_min + 0.5 * (1 + math.cos((t - t_warm) / (t_end - t_warm) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min
    

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float
) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    
    l2_norm = 0
    for g in grads:
        l2_norm += (g.data ** 2).sum()
    l2_norm = l2_norm ** 0.5
    
    alpha = max_l2_norm / (l2_norm + eps)
    
    if l2_norm > max_l2_norm:
        for g in grads:
            g.data *= alpha