import torch
from torch import nn

class RoPE(nn.Module):
    """
    旋转位置编码（Rotary Position Embedding）模块。
    
    通过旋转矩阵对 query 和 key 的向量进行位置相关的变换，使得内积结果天然包含相对位置信息。
    该实现预先计算所有位置的 cos 和 sin 值，并在前向传播中根据 `token_positions` 直接索引取值。
    """
    
    def __init__(
        self, 
        theta: float,
        d_k: int,               # 即 d_model，必须为偶数
        max_seq_len: int,
        device: torch.device | None = None
    ):
        """
        构造 RoPE 模块，预计算所有位置的正弦/余弦值。

        Args:
            theta (float): 基值，用于控制旋转频率的衰减速度，通常取 10000.0。
            d_k (int): 特征维度，必须为偶数（因为每两个维度构成一个旋转对）。
            max_seq_len (int): 最大序列长度，预计算表的大小。
            device (torch.device | None): 张量存放设备，若为 None 则使用默认设备。
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even."
        # 计算逆频率因子，形状 (d_k // 2)
        # 对于第 i 对维度（i=0,1,...,d_k//2-1），频率为 theta^{-2i/d_k}
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # 生成所有位置的位置索引向量 (max_seq_len,)
        positions = torch.arange(max_seq_len)
        # 外积得到每个位置在每个频率下的角度： (max_seq_len, d_k//2)
        freqs = torch.einsum('i, j -> ij', positions, inv_freq)
        # 预计算 cos 和 sin 并注册为 buffer（非持久化，不存入 state_dict）
        self.register_buffer('cos', freqs.cos(), persistent=False)  # (max_seq_len, d_k//2)
        self.register_buffer('sin', freqs.sin(), persistent=False)  # (max_seq_len, d_k//2)

        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        对输入张量 x 应用旋转位置编码。

        假设 x 的形状为 (batch, seq_len, d_k) 或更高维，但最后一维必须为 d_k。
        根据 token_positions 中每个 token 的实际位置取出对应的 cos/sin 值，
        然后对每两个连续维度（构成一个复数对）应用旋转矩阵。

        Args:
            x (torch.Tensor): 输入张量，形状为 (..., d_k)，其中 d_k 为特征维度。
            token_positions (torch.Tensor): 每个 token 在序列中的绝对位置，形状为 (batch, seq_len) 或可广播至该形状。

        Returns:
            torch.Tensor: 旋转后的张量，形状与 x 相同。
        """
        # 1. 根据 token_positions 索引预计算的 cos/sin 表
        #    结果形状为 (batch, seq_len, d_k//2)
        cos = self.cos[token_positions]   # 支持高级索引，要求 token_positions 在 [0, max_seq_len) 内
        sin = self.sin[token_positions]
        # 2. 将最后一维 d_k 拆分为 (d_k//2, 2)，便于成对处理
        #    x_reshaped 形状: (..., d_k//2, 2)
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        # 3. 分别取出每一对的第一个和第二个元素
        x1 = x_reshaped[..., 0]   # (..., d_k//2)
        x2 = x_reshaped[..., 1]   # (..., d_k//2)
        # 4. 应用二维旋转矩阵：
        #    [ x1' ]   [ cos  -sin ] [ x1 ]
        #    [ x2' ] = [ sin   cos ] [ x2 ]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        # 5. 将旋转后的两个分量重新堆叠为最后一维
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)  # (..., d_k//2, 2)
        # 6. 恢复原始形状 (..., d_k)
        return rotated.view(*x.shape)