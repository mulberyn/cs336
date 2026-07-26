import torch
from torch import nn


class Embedding(nn.Module):
    """
    词嵌入模块（Embedding Layer）。

    将整数词元 ID 映射为稠密的向量表示。该类实现了一个可训练的查找表，
    输入为形状 `(batch_size, sequence_length)` 的整数张量，输出为形状
    `(batch_size, sequence_length, embedding_dim)` 的浮点张量。

    权重矩阵的维度为 `(num_embeddings, embedding_dim)`，并采用截断正态分布初始化。

    Attributes:
        weight (nn.Parameter): 嵌入矩阵，形状为 (num_embeddings, embedding_dim)，
                               在训练过程中会被优化。
    """


    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化嵌入模块。

        Args:
            num_embeddings (int): 词表大小（不同 ID 的个数）。
            embedding_dim (int): 每个嵌入向量的维度，即论文中的 d_model。
            device (torch.device | None, optional): 参数存储的设备（如 'cuda'）。
            dtype (torch.dtype | None, optional): 参数的数据类型（如 torch.float32）。
        """
        super().__init__()
        # 创建一个形状为 (num_embeddings, embedding_dim) 的空张量作为参数
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # 使用截断正态分布初始化权重：均值 0，标准差 1，截断范围 [-3, 3]
        # 这使得初始嵌入向量的各维呈标准正态分布，有助于模型早期训练的稳定性
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        执行嵌入查找。（执行词嵌入）

        根据输入的整数 ID 张量，从嵌入矩阵中取出对应行的向量，并返回形状
        为 `(batch_size, sequence_length, embedding_dim)` 的张量。

        Args:
            token_ids (torch.Tensor): 整数 ID 张量，形状为 (batch_size, sequence_length)，
                                      数据类型应为 torch.long。

        Returns:
            torch.Tensor: 嵌入向量张量，形状为 (batch_size, sequence_length, embedding_dim)。
                          数据类型与 self.weight 一致。
        """
        # 利用 PyTorch 的高级索引功能，将 token_ids 中的每个整数视为行索引，
        # 从 self.weight 中取出行向量，输出形状自动增加最后一维 embedding_dim
        return self.weight[token_ids]