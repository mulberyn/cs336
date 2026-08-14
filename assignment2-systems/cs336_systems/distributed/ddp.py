import torch
import torch.distributed as dist

class NaiveDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._handles = []          # 存储 (handle, grad) 元组
        self._registered_ids = set()  # 记录已注册钩子的参数 id

        # 1. 从 rank 0 广播初始参数
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # 2. 为每个唯一可训练参数注册钩子
        for p in self.module.parameters():
            if p.requires_grad:
                pid = id(p)
                if pid not in self._registered_ids:
                    self._registered_ids.add(pid)
                    p.register_hook(self._make_hook())

    def _make_hook(self):
        """返回钩子函数：异步 all-reduce SUM"""
        def hook(grad):
            handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append((handle, grad))
            return grad
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """等待所有异步 all-reduce 完成，然后原地除以 world_size"""
        if not self._handles:
            return
        world_size = dist.get_world_size()
        for handle, grad in self._handles:
            handle.wait()
            grad.div_(world_size)
        self._handles.clear()