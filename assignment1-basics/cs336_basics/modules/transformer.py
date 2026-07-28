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


from .embedding import Embedding
from .linear import Linear
from .softmax import softmax


class TransformerLM(nn.Module):
    """
    基于 Transformer 的自回归语言模型。

    该模块将词元 ID 序列作为输入，并输出下一个词元的未归一化 logits 预测。
    其结构遵循标准的 GPT 式架构：
        1. 可学习的词元嵌入层（Token Embedding），将整数 ID 映射为稠密向量。
        2. 堆叠 `num_layers` 个前置归一化（Pre-Norm）Transformer 块，
           每个块包含因果多头自注意力和 SwiGLU 前馈网络，并内置 RoPE 位置编码。
        3. 最终的 RMSNorm 层，稳定输出特征。
        4. 线性输出头（LM Head），将特征映射回词表大小，生成 logits。

    整个模型采用因果掩码，确保训练时不会关注未来词元，从而保持自回归特性。

    Attributes:
        token_embedding (Embedding): 词元嵌入层，将整数 ID 转换为稠密向量。
        transformer_blocks (nn.ModuleList): 包含 `num_layers` 个 Transformer 块的列表。
        output_norm (RMSNorm): 应用于最终输出的 RMSNorm。
        output_embedding (Linear): 输出投影层，将隐藏状态映射为词表大小的 logits。
    """

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
        """
        初始化 Transformer 语言模型。

        Args:
            vocab_size (int): 词表大小，决定了嵌入矩阵和输出头的维度。
            context_length (int): 最大上下文长度（即最大序列长度），
                                  用于预计算 RoPE 的正余弦表。
            d_model (int): 模型的隐藏维度，即嵌入向量和子层输出的维度。
            num_layers (int): Transformer 块的堆叠数量。
            num_heads (int): 多头注意力中的头数。`d_model` 必须能被 `num_heads` 整除。
            d_ff (int): 前馈网络内部层的维度（SwiGLU 的中间维度）。
            rope_theta (float): RoPE 的基础常数，通常为 10000.0。
            device (torch.device | None, optional): 所有参数的存储设备。
            dtype (torch.dtype | None, optional): 所有参数的数据类型。
        """
        super().__init__()

        # 1. 词元嵌入层：将整数 ID 映射为 d_model 维向量
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # 2. 堆叠 N 个 Transformer 块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,  # 传递给 RoPE
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        # 3. 最终输出归一化
        self.output_norm = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)

        # 4. 输出投影头（LM Head）：将 d_model 映射为 vocab_size（生成 logits）
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
        """
        执行语言模型的前向传播。

        处理流程：
            1. 将输入的整数 ID 通过嵌入层转换为稠密向量。
            2. 依次通过所有 Transformer 块（每个块内部自动处理因果掩码和 RoPE）。
            3. 对最终输出应用 RMSNorm。
            4. 通过线性层投影为词表大小的 logits。

        注意：
            - 该实现返回的是 **未归一化的 logits**，而不是概率分布。
            - 这与 PyTorch 的 `CrossEntropyLoss` 兼容，该损失函数内部会自动应用 Softmax。
            - 若需要概率分布，可在外层对 logits 应用 `F.softmax(dim=-1)`。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length)，
                              数据类型为 `torch.long`（整数 ID）。

        Returns:
            torch.Tensor: 输出 logits 张量，形状为 (batch_size, sequence_length, vocab_size)，
                          数据类型为 `torch.float`。
        """
        # 嵌入：将整数 ID 转换为向量
        x = self.token_embedding(x)  # (batch, seq_len, d_model)

        # 逐层通过 Transformer 块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # (batch, seq_len, d_model)

        # 最终归一化
        x = self.output_norm(x)  # (batch, seq_len, d_model)

        # 输出投影：生成 logits（注意：此处不进行 Softmax）
        x = self.output_embedding(x)  # (batch, seq_len, vocab_size)

        return x