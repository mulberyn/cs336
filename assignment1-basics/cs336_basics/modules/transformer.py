import torch
from torch import nn

from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiHeadSelfAttention
from .rope import RoPE
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
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
        
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x