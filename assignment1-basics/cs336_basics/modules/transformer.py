import torch
from torch import nn

from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiHeadSelfAttention
from .rope import RoPE
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    前置归一化（Pre‑Norm）Transformer 块。

    该模块将多头自注意力（MHA）和 SwiGLU 前馈网络（FFN）组合成一个标准的
    Transformer 层，并采用 Pre‑Norm 结构：每个子层先进行 RMSNorm，
    然后执行主操作，最后通过残差连接将输入与输出相加。

    结构遵循现代大语言模型（如 LLaMA、GPT‑3）的常见设计：
        - 第一个子层：RMSNorm → MultiHeadSelfAttention → 残差连接
        - 第二个子层：RMSNorm → SwiGLU FFN → 残差连接

    整个模块不包含任何可学习的额外参数，所有参数均由其子模块（MHA、FFN、RMSNorm）提供。

    Attributes:
        norm1 (RMSNorm): 应用于注意力子层之前的 RMSNorm。
        norm2 (RMSNorm): 应用于前馈子层之前的 RMSNorm。
        mha (MultiHeadSelfAttention): 多头自注意力模块（已注入 RoPE）。
        ffn (SwiGLU): SwiGLU 前馈网络模块。
    """

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
        """
        初始化 Transformer 块。

        Args:
            d_model (int): 输入和输出的特征维度。
            num_heads (int): 多头注意力的头数。`d_model` 必须能被 `num_heads` 整除。
            d_ff (int): 前馈网络内部层的维度（通常为 `8/3 * d_model`，但此处由调用者指定）。
            max_seq_len (int): 最大序列长度，用于预计算 RoPE 的正余弦表。
            theta (float): RoPE 的基础常数（通常为 10000.0）。
            device (torch.device | None, optional): 所有参数的存储设备。
            dtype (torch.dtype | None, optional): 所有参数的数据类型。
        """
        super().__init__()

        # 两个独立的 RMSNorm（参数不共享）
        self.norm1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)

        # 创建 RoPE 模块并注入到 MHA
        d_k = d_model // num_heads
        rope = RoPE(theta, d_k, max_seq_len, device=device, dtype=dtype)

        self.mha = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoding=rope,
            device=device,
            dtype=dtype
        )

        # SwiGLU 前馈网络
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        执行 Transformer 块的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, d_model)。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同 (batch_size, sequence_length, d_model)。
        """
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x