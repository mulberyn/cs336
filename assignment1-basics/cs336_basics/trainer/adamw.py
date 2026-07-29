import torch
from torch.optim import Optimizer
from collections.abc import Callable
from typing import Optional


class AdamW(Optimizer):
    """
    AdamW 优化器（带权重衰减解耦的 Adam 变体）。

    该实现遵循 Algorithm 2 中的 AdamW 伪代码（Loshchilov & Hutter, 2019）。
    与原始 Adam 相比，AdamW 将权重衰减与梯度更新解耦，从而在自适应学习率设置下
    实现更有效的正则化。

    对于每个参数，优化器维护：
        - 一阶矩估计 m（动量）
        - 二阶矩估计 v（平方梯度的指数移动平均）
        - 迭代步数 t
    """
    
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float
    ):
        assert lr > 0, "learning rate must be greater than 0."
        defaults = {
            'lr': lr,
            'beta1': betas[0],
            'beta2': betas[1],
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)
        
    
    def step(
        self, 
        closure: Optional[Callable] = None
    ):
        """
        执行单步参数更新。

        该函数按以下顺序执行：
            1. 如果提供了 `closure`，则调用它以重新计算损失。
            2. 遍历每个参数组，对其中的每个参数执行 AdamW 更新。

        Args:
            closure (Optional[Callable], optional): 一个重新计算损失的可调用对象。
                通常用于需要多次前向-反向传播的复杂场景。默认为 None。

        Returns:
            loss (optional): 如果提供了 closure，则返回其计算出的损失，否则返回 None。
        """
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                lr, beta1, beta2, weight_decay, eps = group['lr'], group['beta1'], group['beta2'], group['weight_decay'], group['eps']
                m, v, step = state['m'], state['v'], state['step']
                
                step += 1
                state['step'] = step
                
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                state['m'] = m
                state['v'] = v
                
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                
                p.data -= lr * weight_decay * p.data
                p.data -= lr * m_hat / (v_hat ** 0.5 + eps)
                
        return loss