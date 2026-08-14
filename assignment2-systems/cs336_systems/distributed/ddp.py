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
