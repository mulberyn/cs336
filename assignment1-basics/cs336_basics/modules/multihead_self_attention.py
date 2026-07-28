import torch
from torch import nn
from einops import rearrange

from .linear import Linear
from .scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力模块（支持因果掩码和可插拔位置编码）。

    该模块实现了 Transformer 中的多头自注意力机制，并默认应用因果掩码（下三角掩码），
    确保序列中每个位置只能关注到它自身及其之前的位置（自回归属性）。

    它支持通过依赖注入的方式传入位置编码模块（如 RoPE），使得位置编码的插入更加灵活，
    且仅对 Query 和 Key 进行位置编码，不影响 Value。

    Attributes:
        d_model (int): 模型的输入/输出特征维度。
        num_heads (int): 注意力头的数量。
        d_k (int): 每个注意力头的特征维度，等于 d_model // num_heads。
        wq (Linear): Query 的线性投影层。
        wk (Linear): Key 的线性投影层。
        wv (Linear): Value 的线性投影层。
        wo (Linear): 输出投影层。
        pos_enc (nn.Module | None): 可插拔的位置编码模块（如 RoPE）。
    """
    
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
        """
        执行多头自注意力的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (..., seq_len, d_model)，
                其中 ... 表示任意数量的前导批次维度。
            token_positions (torch.Tensor | None, optional): 每个 token 对应的位置索引，
                形状为 (..., seq_len) 或 (seq_len,)。
                如果为 None，则默认使用 0 到 seq_len-1 的连续位置。
                仅在启用位置编码时有效。

        Returns:
            torch.Tensor: 注意力输出张量，形状为 (..., seq_len, d_model)，
                与输入张量的形状完全相同（仅数值改变）。
        """
        seq_len = x.size(-2)
        # 1. 线性投影（形状不变） 
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        # 2. 拆分为多头 (..., num_heads, seq_len, d_k) 
        q = rearrange(q, '... s (n_h d_k) -> ... n_h s d_k', n_h=self.num_heads) # 将 n_h 提取到 s 前面
        k = rearrange(k, '... s (n_h d_k) -> ... n_h s d_k', n_h=self.num_heads)
        v = rearrange(v, '... s (n_h d_k) -> ... n_h s d_k', n_h=self.num_heads)
        
        # 3. 应用 RoPE（仅 Q 和 K） 
        # 若未提供 token_positions，默认使用 0..seq_len-1
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        if self.pos_enc is not None:
            q = self.pos_enc(q, token_positions)
            k = self.pos_enc(k, token_positions)
            
        # 4. 构造因果掩码（下三角 (seq_len, seq_len)，True 表示可见） 
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        
        # 5. 缩放点积注意力 
        attn = scaled_dot_product_attention(q, k, v, causal_mask)
        
        # 6. 多头合并 
        attn = rearrange(attn, '... n_h s d_k -> ... s (n_h d_k)', n_h=self.num_heads)
        
        # 7. 输出投影 
        out = self.wo(attn)
        
        return out