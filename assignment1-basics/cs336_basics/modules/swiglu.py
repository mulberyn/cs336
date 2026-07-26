import torch
from torch import nn
from .linear import Linear
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络模块（SwiGLU Feed-Forward Network）。

    该模块实现了现代大语言模型（如 LLaMA、Qwen）中常用的 SwiGLU 激活函数，
    它结合了 SiLU（Swish）激活和门控线性单元（GLU），并在线性层中省略了偏置项。

    数学公式：
        FFN(x) = W2 * (SiLU(W1 * x) ⊙ (W3 * x))
    其中：
        - W1, W3 : 升维线性变换，形状为 (d_ff, d_model)
        - W2     : 降维线性变换，形状为 (d_model, d_ff)
        - SiLU(z) = z * sigmoid(z)
        - ⊙      : 逐元素相乘（门控机制）

    该模块首先将输入从 d_model 维升维到 d_ff 维，通过两条支路（激活支路和门控支路）
    进行非线性变换和门控，最后投影回 d_model 维。中间维度 d_ff 通常取 8/3 * d_model，
    并向上取整为 64 的倍数，以充分利用硬件性能。

    Attributes:
        w1 (Linear): 激活支路的线性层，将 d_model 映射到 d_ff。
        w2 (Linear): 投影层的线性层，将 d_ff 映射回 d_model。
        w3 (Linear): 门控支路的线性层，将 d_model 映射到 d_ff。
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化 SwiGLU 前馈网络。

        Args:
            d_model (int): 输入和输出的特征维度（即模型维度）。
            d_ff (int, optional): 内部前馈层的维度。若未指定，则自动设为
                                   (8/3) * d_model 并向上取整为 64 的倍数。
            device (torch.device | None, optional): 参数存储设备。
            dtype (torch.dtype | None, optional): 参数数据类型。
        """
        super().__init__()

        # 若未指定 d_ff，则按 8/3 * d_model 计算并对齐到 64 的倍数
        if d_ff is None:
            d_ff = int((8 / 3) * d_model)
            d_ff = ((d_ff + 63) // 64) * 64

        # 激活支路：d_model -> d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # 门控支路：d_model -> d_ff
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        # 投影层：d_ff -> d_model
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 SwiGLU 前向传播。

        处理流程：
            1. 将输入 x 分别通过 w1 和 w3 升维到 d_ff 维。
            2. 对 w1 的输出应用 SiLU 激活函数（即 x * sigmoid(x)）。
            3. 将激活结果与 w3 的输出逐元素相乘（门控机制）。
            4. 通过 w2 投影回 d_model 维。

        Args:
            x (torch.Tensor): 输入张量，形状为 (..., d_model)，
                               其中 ... 表示任意批次维度（如 batch_size, sequence_length）。

        Returns:
            torch.Tensor: 输出张量，形状为 (..., d_model)，与输入形状相同（仅最后一维不变）。
        """
        # 公式：W2 * (SiLU(W1 * x) ⊙ (W3 * x))
        # F.silu 是 PyTorch 提供的 SiLU 实现，数值稳定且高效
        return self.w2(F.silu(self.w1(x)) * self.w3(x))