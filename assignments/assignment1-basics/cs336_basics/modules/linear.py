import torch
from torch import nn
import math

class Linear(nn.Module):
    """
    自定义线性层（全连接层），无偏置项。
    
    数学公式: y = x @ W
    权重形状: (in_features, out_features)
    
    遵循 GPT/LLaMA 等现代 LLM 的惯例，不包含偏置（bias）参数。
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        初始化线性层的权重参数。

        Args:
            in_features: 输入特征维度（最后一维的大小）
            out_features: 输出特征维度（最后一维的大小）
            device: 参数存储设备（如 'cuda' 或 'cpu'）
            dtype: 参数数据类型（如 torch.float32）
        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return x @ self.weight