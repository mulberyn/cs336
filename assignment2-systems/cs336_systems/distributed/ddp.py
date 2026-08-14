import torch
import torch.distributed as dist


class NaiveDDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        # synchronize initial parameters
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class DDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()
        # synchronize initial parameters
        seen = set()
        for p in self.module.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            dist.broadcast(p.data, src=0)

        # register backward hooks
        seen.clear()
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            p.register_post_accumulate_grad_hook(
                self._make_hook()
            )
    

    def _make_hook(self):
        def hook(param):
            if param.grad is None:
                return
            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            self.handles.append((handle, param.grad))
        return hook
    

    def finish_gradient_synchronization(self):
        for handle, grad in self.handles:
            handle.wait()
            grad.div_(self.world_size)
        self.handles.clear()
    

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)