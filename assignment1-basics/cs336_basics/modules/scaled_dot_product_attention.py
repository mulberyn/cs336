from .softmax import softmax
import torch


def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    计算缩放点积注意力（Scaled Dot-Product Attention）。

    根据 Transformer 论文中的公式：
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    该实现使用 einsum 自动处理键矩阵 K 的转置，并支持任意数量的前导批次维度
    （如 batch_size、num_heads 等）。同时支持可选的掩码机制，用于屏蔽未来信息
    或填充位置。

    Args:
        Q (torch.Tensor): 查询张量，形状为 (..., seq_q, d_k)。
            '...' 表示任意数量的前导批次维度。
        K (torch.Tensor): 键张量，形状为 (..., seq_k, d_k)。
            seq_k 通常与 seq_q 相同（自注意力），但在交叉注意力中可以不同。
        V (torch.Tensor): 值张量，形状为 (..., seq_k, d_v)。
            seq_k 必须与 K 的序列长度一致。
        mask (torch.Tensor | None, optional): 布尔掩码张量，形状为
            (seq_q, seq_k) 或 (1, 1, seq_q, seq_k)。
            - True  : 表示该位置允许参与注意力计算（信息流动）。
            - False : 表示该位置被屏蔽，对应的注意力权重将被置为 0。
            如果传入的是 2D 掩码，会自动广播到所有批次维度和注意力头。
            默认为 None（不使用掩码）。

    Returns:
        torch.Tensor: 注意力输出张量，形状为 (..., seq_q, d_v)。
            与 Q 的前导批次维度保持一致，最后一维变为 d_v。

    Note:
        - 缩放因子 sqrt(d_k) 用于防止点积结果过大，避免 softmax 梯度进入饱和区。
        - 掩码实现技巧：对于 mask=False 的位置，在 softmax 之前将对应分数设为 -inf，
          使得 exp(-inf) = 0，从而在归一化后权重为 0。
        - 当前实现使用 torch.einsum 计算 Q 和 K 的点积，无需显式调用 .transpose()，
          代码更简洁且与数学公式对齐。
    """
    # 1. 计算缩放前的注意力分数（einsum 自动处理 K 的转置）
    # '... q d, ... k d -> ... q k' 在 d 维度上做点积，q 和 k 维度保留
    attn = torch.einsum('... q d, ... k d -> ... q k', Q, K) / (Q.size(-1) ** 0.5)
    # 2. 应用掩码（如果提供）：将 False 位置填充为 -inf
    if mask is not None:
        # 注意：masked_fill 在条件为 True 的位置填充，而我们要填充 False 的位置，
        # 所以使用取反操作 ~mask
        attn = attn.masked_fill(~mask, -torch.inf)
    # 3. 在最后一个维度（seq_k）上执行 softmax，得到注意力权重（概率分布）
    # 4. 将注意力权重与值矩阵 V 相乘，得到最终输出
    output = softmax(attn, dim=-1) @ V
    return output