import torch
from torch import nn


class RMSnorm(nn.Module):
    """
    均方根层归一化（Root Mean Square Layer Normalization）。

    数学公式：
        RMSNorm(a_i) = (a_i / RMS(a)) * g_i
    其中：
        RMS(a) = sqrt( mean(a^2) + eps )
        g_i 是可学习的增益参数，形状为 (d_model,)

    输入形状为 (..., d_model) 的张量，输出形状保持不变。所有位置共享同一组增益参数。

    Attributes:
        eps (float): 防止分母为零的小常数，默认 1e-5。
        weight (nn.Parameter): 可学习的增益参数，形状 (d_model,) **初始化为全 1**。(handout 中的 g 向量)
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化 RMSNorm 模块。

        Args:
            d_model (int): 特征维度（即每个令牌的向量长度）。
            eps (float, optional): 添加到分母的微小常数，防止除零。默认为 1e-5。
            device (torch.device | None, optional): 参数存储设备（如 'cuda'）。默认为 None。
            dtype (torch.dtype | None, optional): 参数数据类型（如 torch.float32）。默认为 None。
        """
        super().__init__()
        self.eps = eps
        # 增益参数初始化为全 1，使得模块初始时近似恒等映射
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用 RMSNorm。

        处理流程：
            1. 将输入提升至 torch.float32 以防止平方溢出（特别适用于混合精度训练）。
            2. 计算最后一个维度（d_model）上的均方根（RMS）。
            3. 用 RMS 归一化输入。
            4. 乘以可学习的增益参数。
            5. 将结果转回原始数据类型以节省显存。

        Args:
            x (torch.Tensor): 输入张量，形状为 (..., d_model)，其中 ... 表示任意批次维度。

        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同 (..., d_model)，数据类型与输入一致。
        """
        # 保存输入原始数据类型
        in_dtype = x.dtype
        # 转为 float32 以保证数值稳定性（避免 float16 平方溢出）
        x = x.to(torch.float32)

        # 计算均方根：sqrt(mean(x^2) + eps)，keepdim=True 保留维度便于广播
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # 归一化
        x_normed = x / rms

        # 乘以可学习的增益参数
        result = x_normed * self.weight

        # 恢复原始数据类型（如 float16/bfloat16）以节省内存
        return result.to(in_dtype)