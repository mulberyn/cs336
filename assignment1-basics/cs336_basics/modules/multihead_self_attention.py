import torch
from torch import nn
from einops import rearrange

from .linear import Linear
from .scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoding: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 四个线性投影
        self.wq = Linear(d_model, d_model, device, dtype)
        self.wk = Linear(d_model, d_model, device, dtype)
        self.wv = Linear(d_model, d_model, device, dtype)
        self.wo = Linear(d_model, d_model, device, dtype)
        
        # RoPE 模块（共用，作用于每个头的 d_k 维）
        self.pos_enc = positional_encoding
        
    
    def forward(
        self,
        x: torch.Tensor, 
        token_positions: torch.Tensor | None = None
    ):
        seq_len = x.size(-2)
        # ----- 1. 线性投影（形状不变） -----
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        # ----- 2. 拆分为多头 (..., num_heads, seq_len, d_k) -----
        q = rearrange(q, '... s (n_h d_k) -> ... n_h s d_k', n_h=self.num_heads) # 将 n_h 提取到 s 前面
        k = rearrange(k, '... s (n_h d_k) -> ... n_h s d_k', n_h=self.num_heads)
        v = rearrange(v, '... s (n_h d_k) -> ... n_h s d_k', n_h=self.num_heads)
        
        # ----- 3. 应用 RoPE（仅 Q 和 K） -----
        # 若未提供 token_positions，默认使用 0..seq_len-1
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        if self.pos_enc is not None:
            q = self.pos_enc(q, token_positions)
            k = self.pos_enc(k, token_positions)
            
        # ----- 4. 构造因果掩码（下三角 (seq_len, seq_len)，True 表示可见） -----
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        
        # ----- 5. 缩放点积注意力 -----
        attn = scaled_dot_product_attention(q, k, v, causal_mask)
        
        # ----- 6. 多头合并 -----
        attn = rearrange(attn, '... n_h s d_k -> ... s (n_h d_k)', n_h=self.num_heads)
        
        # ----- 7. 输出投影 -----
        out = self.wo(attn)
        
        return out