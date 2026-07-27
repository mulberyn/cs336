import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(
        self, 
        theta: float,
        d_k: int, # 即 d_model
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even."
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        positions = torch.arange(max_seq_len)
        freqs = torch.einsum('i, j -> ij', positions, inv_freq) # (max_seq_len, d_k//2)
        # 通过得到的 freqs，计算对应的 cos/sin 值并存下来       
        self.register_buffer('cos', freqs.cos(), persistent=False) # (max_seq_len, d_k//2)
        self.register_buffer('sin', freqs.sin(), persistent=False) # (max_seq_len, d_k//2)

        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        # token_positions 是 (batch, seq_len)，直接索引 cos[token_positions] -> (batch, seq_len, d_k//2)
        cos = self.cos[token_positions]  # 支持高级索引
        sin = self.sin[token_positions]
        # 分别取出 (..., d_k // 2, 2) 的最后一维的两个数字，变成 (..., d_k // 2)
        # shape[:-1] 取出除了 d_k 这一维，-1 表示剩余维度，2 表示将最后一维固定为 2
        x_reshaped = x.view(*x.shape[:-1], -1, 2) # (..., d_k) -> (..., d_ // 2, 2)
        x1 = x_reshaped[..., 0] 
        x2 = x_reshaped[..., 1]
        # 旋转
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        # 拼接最后一维，变成新的张量，两个 (..., d_k // 2) 在最后一维合并成 (..., d_k // 2, 2)
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return rotated.view(*x.shape)