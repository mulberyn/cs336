import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    """
    朴素 DDP 容器，负责：
    - 初始化时从 rank 0 广播模型参数
    - 为每个需要梯度的参数注册 backward hook，异步 all-reduce 其梯度
    - 提供 finish_gradient_synchronization() 等待所有异步操作完成
    """
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._async_ops = []   # 存储未完成的异步 all-reduce 操作

        # 1. 将 rank 0 的模型参数广播到所有其他进程
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 2. 为每个可训练参数注册 backward hook
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook(grad):
            # 异步 all-reduce 平均梯度
            handle = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
            self._async_ops.append(handle)
            return grad   # 返回修改后的梯度（原地修改，异步完成后生效）
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """等待所有未完成的梯度同步操作完成，并清空记录"""
        for handle in self._async_ops:
            handle.wait()
        self._async_ops.clear()