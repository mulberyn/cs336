import torch

def softmax(
    x: torch.Tensor,
    dim: int
):
    # 假设 max(dim=1, keepdim=True) 就得到 (B, 1, D)
    max_val = x.max(dim=dim, keepdim=True)[0]
    x -= max_val
    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp