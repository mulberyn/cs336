import copy
import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls,
        **kwargs,
    ):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.wrapped_optimizer = None
        super().__init__(params, {})

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        local_group = copy.deepcopy(param_group)
        local_params = []
        for idx, p in enumerate(param_group["params"]):
            if idx % self.world_size == self.rank:
                local_params.append(p)
        local_group["params"] = local_params
        if self.wrapped_optimizer is None:
            self.wrapped_optimizer = self.optimizer_cls(
                [local_group],
                **self.optimizer_kwargs,
            )
        else:
            self.wrapped_optimizer.add_param_group(local_group)

    def step(self, closure=None, **kwargs):
        loss = self.wrapped_optimizer.step(
            closure=closure,
            **kwargs,
        )
        index = 0
        for group in self.param_groups:
            for p in group["params"]:
                owner = index % self.world_size
                dist.broadcast(
                    p.data,
                    src=owner,
                )
                index += 1
        return loss