import torch
import timeit
import sys

from typing import TypedDict
from typing import Literal

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

TrainingType = Literal["forward-only", "forward-and-backwrad", "whole-steps"]
ModelScale = Literal["xxs", "xs", "s", "m", "l", "xl"]

MODELS_CONFIGS = {
    "xxs": {
        "num_layers": 2,
        "d_model": 256,
        "d_ff": 1024, 
        "num_heads": 4,
    },
    "xs": {
        "num_layers": 4,
        "d_model": 512, 
        "d_ff": 2048,
        "num_heads": 8,
    },
    "s": {
        "num_layers": 8,
        "d_model": 768,
        "d_ff": 3072,
        "num_heads": 12,
    },
    "m": {
        "num_layers": 16,
        "d_model": 1024,
        "d_ff": 4096,
        "num_heads": 16,
    },
    "l": {
        "num_layers": 24,
        "d_model": 1536,
        "d_ff": 6144,
        "num_heads": 16,
    },
    "xl": {
        "num_layers": 8,
        "d_model": 32,
        "d_ff": 2560,
        "num_heads": 10240,
    },
}

class DefaultModel(TypedDict):
    model_scale: ModelScale
    train_types: TrainingType
    context_length: 
    warm_steps: int
    count_steps: int


DEFAULT_MODEL: DefaultModel = {
    "model_scale": "xxs",
    "train_types": "whole-steps",
    "context_length": 256,
    "warm_steps": 12,
    "count_steps": 50,
    "device": "cuda"
}

from numpy.typing import NDArray
import numpy as np


def run(
    training_type: TrainingType,
    model_scale: ModelScale,
    context_length: int,
    warm_steps: int,
    count_steps: int,
    device,
):
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=MODELS_CONFIGS[training_type]["d_model"],
        num_layers=MODELS_CONFIGS[training_type]["num_layers"],
        num_heads=MODELS_CONFIGS[training_type]["num_heads"],
        d_ff=MODELS_CONFIGS[training_type]["d_ff"],
    )
    
    opt = AdamW(
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    )
    
    batch_size = torch.randint(0, 10).item()
    dataset: NDArray[np.int_] = np.random.randint(0, 10000, size=context_length * batch_size // 2)
    


def main():
    

if __name__ == "__main__":
    sys.exit(main())