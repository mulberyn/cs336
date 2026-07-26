import torch
from torch import nn
from .linear import Linear
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int, 
        d_ff: int = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = int((8 / 3) * d_model)
            d_ff = ((d_ff + 63) // 64) * 64
        
        self.w1 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))