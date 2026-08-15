from __future__ import annotations
from typing import Any
import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from cs336_basics.model import Linear as CS336Linear
    from cs336_basics.model import Embedding as CS336Embedding
except ImportError:
    CS336Linear = nn.Linear
    CS336Embedding = nn.Embedding
_SHARDED_TYPES = (
    nn.Linear,
    nn.Embedding,
    CS336Linear,
    CS336Embedding,
)

class FSDP(nn.Module):
    """
    A simplified Fully Sharded Data Parallel implementation.

    Design:

    - Linear / Embedding weights are sharded along dimension 0.
    - Replicated parameters (e.g. RMSNorm weights) remain replicated.
    - Forward:
        local shard -> all_gather -> temporary full weight -> forward.
    - Backward:
        full weight gradient -> reduce_scatter -> local gradient.
    - Master parameters remain in FP32.
    - compute_dtype is used for communication and computation.
    """
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._pending: list[tuple[Any, ...]] = []
        self._sharded_params: dict[int, dict[str, Any]] = {}
        # ------------------------------------------------------------------
        # 1. Synchronize initial model parameters.
        # ------------------------------------------------------------------
        seen = set()
        for p in self.module.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            dist.broadcast(
                p.data,
                src=0,
            )
        # ------------------------------------------------------------------
        # 2. Shard Linear / Embedding weights.
        # ------------------------------------------------------------------
        for submodule in self.module.modules():
            if not isinstance(submodule, _SHARDED_TYPES):
                continue
            if not hasattr(submodule, "weight"):
                continue
            weight = submodule.weight
            if not weight.requires_grad:
                continue
            self._shard_weight(submodule)
        # ------------------------------------------------------------------
        # 3. Register hooks for replicated parameters.
        #
        # These are parameters such as RMSNorm.weight.
        # ------------------------------------------------------------------
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            if id(p) in self._sharded_params:
                continue
            # 这个参数 backward 产生 gradient 后，自动进行 all-reduce。
            p.register_post_accumulate_grad_hook(
                self._make_replicated_grad_hook()
            )
    # ----------------------------------------------------------------------
    # Parameter sharding
    # ----------------------------------------------------------------------
    def _shard_weight(self, module: nn.Module) -> None:
        original_weight = module.weight
        original_shape = tuple(original_weight.shape)
        if original_weight.ndim == 0:
            return
        num_rows = original_shape[0]
        local_rows = (num_rows + self.world_size - 1) // self.world_size
        padded_rows = local_rows * self.world_size
        # Master weights stay FP32.
        full_weight = original_weight.data.to(torch.float32)
        # Pad if the first dimension is not divisible by world_size.
        if padded_rows != num_rows: #有行空出来了
            padded_shape = (
                padded_rows,
                *original_shape[1:],
            )
            padded_weight = torch.zeros( # 先全初始化为 0
                padded_shape,
                dtype=full_weight.dtype,
                device=full_weight.device,
            )
            padded_weight[:num_rows].copy_(full_weight)
            full_weight = padded_weight
        # Ensure all ranks have exactly the same master weight.
        dist.broadcast(
            full_weight,
            src=0,
        )
        start = self.rank * local_rows
        end = start + local_rows
        local_weight = full_weight[start:end].contiguous()
        local_param = nn.Parameter(
            local_weight,
            requires_grad=original_weight.requires_grad,
        )
        # Replace the original Parameter with its local shard.
        module._parameters["weight"] = local_param
        self._sharded_params[id(local_param)] = {
            "module": module,
            "original_shape": original_shape,
            "num_rows": num_rows,
            "local_rows": local_rows,
            "padded_rows": padded_rows,
        }
        # Forward hooks.
        module.register_forward_pre_hook(
            self._make_forward_pre_hook(module, local_param)
        )
        module.register_forward_hook(
            self._make_forward_post_hook(module, local_param)
        )
    # ----------------------------------------------------------------------
    # Forward: all-gather full weight
    # ----------------------------------------------------------------------
    def _make_forward_pre_hook(
        self,
        module: nn.Module,
        local_param: nn.Parameter,
    ):
        def hook(module: nn.Module, inputs):
            # --------------------------------------------------------------
            # Convert local master weight to compute dtype before communication.
            # --------------------------------------------------------------
            local_weight = local_param.data
            if self.compute_dtype is not None:
                local_weight = local_weight.to(self.compute_dtype)
            # --------------------------------------------------------------
            # All-gather local shards.
            # --------------------------------------------------------------
            shards = [
                torch.empty_like(local_weight)
                for _ in range(self.world_size)
            ]
            dist.all_gather(
                shards,
                local_weight,
            )
            full_padded = torch.cat(
                shards,
                dim=0,
            )
            metadata = self._sharded_params[id(local_param)]
            num_rows = metadata["num_rows"]
            full_weight = nn.Parameter(
                full_padded,
                requires_grad=True,
            )
            # --------------------------------------------------------------
            # Hook full weight gradient.
            # --------------------------------------------------------------
            full_weight.register_post_accumulate_grad_hook(
                self._make_full_weight_grad_hook(
                    local_param,
                    metadata,
                )
            )
            # Save the temporary full parameter.
            module.__dict__["_fsdp_full_weight"] = full_weight
            # Important:
            #
            # We modify the original module's Parameter dictionary instead
            # of replacing the module itself.
            #
            # Therefore:
            #
            # isinstance(module, Linear)
            #
            # remains True.
            module._parameters["weight"] = full_weight
        return hook
    # ----------------------------------------------------------------------
    # Forward post hook: restore local shard
    # ----------------------------------------------------------------------
    def _make_forward_post_hook(
        self,
        module: nn.Module,
        local_param: nn.Parameter,
    ):
        def hook(module: nn.Module, inputs, output):
            # Restore the sharded master parameter.
            module._parameters["weight"] = local_param
        return hook
    # ----------------------------------------------------------------------
    # Full gradient -> reduce scatter
    # ----------------------------------------------------------------------
    def _make_full_weight_grad_hook(
        self,
        local_param: nn.Parameter,
        metadata: dict[str, Any],
    ):
        def hook(full_param: nn.Parameter):
            if full_param.grad is None:
                return
            full_grad = full_param.grad
            # --------------------------------------------------------------
            # Gloo compatibility.
            #
            # Some PyTorch/Gloo versions have limited reduce_scatter
            # support. For the CS336 tests we use Gloo, so fall back to
            # all_reduce + local slice if necessary.
            # --------------------------------------------------------------
            backend = str(dist.get_backend()).lower()
            if backend == "gloo":
                handle = dist.all_reduce(
                    full_grad,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
                self._pending.append(
                    (
                        "sharded_gloo",
                        handle,
                        local_param,
                        full_grad,
                        metadata,
                    )
                )
            else:
                # 当前 full_grad 是根据完整的参数和部分批次数据得到的
                chunks = list(
                    full_grad.chunk(
                        self.world_size,
                        dim=0,
                    )
                )
                local_grad = torch.empty_like(
                    chunks[self.rank]
                )
                handle = dist.reduce_scatter(
                    local_grad,
                    chunks,
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                )
                self._pending.append(
                    (
                        "sharded",
                        handle,
                        local_param,
                        local_grad,
                        metadata,
                    )
                )
        return hook
    # ----------------------------------------------------------------------
    # Replicated parameter gradient hook
    # ----------------------------------------------------------------------
    def _make_replicated_grad_hook(self):
        def hook(param: nn.Parameter):
            if param.grad is None:
                return
            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            self._pending.append(
                (
                    "replicated",
                    handle,
                    param,
                )
            )
        return hook
    # ----------------------------------------------------------------------
    # Wait for all communication
    # ----------------------------------------------------------------------
    def finish_gradient_synchronization(self):
        for item in self._pending:
            kind = item[0]
            # --------------------------------------------------------------
            # Replicated parameter
            # --------------------------------------------------------------
            if kind == "replicated":
                _, handle, param = item
                handle.wait()
                param.grad.div_(self.world_size)
            # --------------------------------------------------------------
            # Sharded parameter, Gloo fallback
            # --------------------------------------------------------------
            elif kind == "sharded_gloo":
                (
                    _,
                    handle,
                    local_param,
                    full_grad,
                    metadata,
                ) = item
                handle.wait()
                full_grad.div_(self.world_size)
                local_rows = metadata["local_rows"]
                start = self.rank * local_rows
                end = start + local_rows
                local_grad = full_grad[
                    start:end
                ]
                local_grad = local_grad.to(
                    local_param.dtype
                )
                if local_param.grad is None:
                    local_param.grad = local_grad.clone()
                else:
                    local_param.grad.add_(
                        local_grad
                    )
            # --------------------------------------------------------------
            # True reduce-scatter
            # --------------------------------------------------------------
            elif kind == "sharded":
                (
                    _,
                    handle,
                    local_param,
                    local_grad,
                    metadata,
                ) = item
                handle.wait()
                local_grad.div_(self.world_size)
                local_grad = local_grad.to(local_param.dtype)
                if local_param.grad is None:
                    local_param.grad = local_grad.clone()
                else:
                    local_param.grad.add_(
                        local_grad
                    )
        self._pending.clear()
    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, *inputs, **kwargs):
        return self.module(
            *inputs,
            **kwargs,
        )
    # ----------------------------------------------------------------------
    # Gather full parameters
    # ----------------------------------------------------------------------
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        result = {}
        for name, param in self.module.named_parameters():
            metadata = self._sharded_params.get(
                id(param)
            )
            # --------------------------------------------------------------
            # Replicated parameter
            # --------------------------------------------------------------
            if metadata is None:
                result[name] = param.data.detach().clone()
                continue
            # --------------------------------------------------------------
            # Sharded parameter
            # --------------------------------------------------------------
            local_weight = param.data
            shards = [
                torch.empty_like(local_weight)
                for _ in range(self.world_size)
            ]
            dist.all_gather(
                shards,
                local_weight,
            )
            full_padded = torch.cat(
                shards,
                dim=0,
            )
            num_rows = metadata["num_rows"]
            original_shape = metadata["original_shape"]
            full_weight = full_padded[
                :num_rows
            ]
            result[name] = full_weight.reshape(
                original_shape
            ).detach().clone()
        return result