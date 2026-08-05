import torch
import timeit
import sys

from typing import TypedDict
from typing import Literal

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

DEFAULT_CONFIG = {
    'vocab_size': 10000,
    'context_length': 256,
    'd_model': 512,
    'num_layers': 4,
    'd_ff': 1344,
    'num_heads': 16,
    'rope_theta': 10000.0,
    
    'max_learning_rate': 1e-3,
    'min_learning_rate': 1e-4,
    'warmup_iters': 500,
    'cosine_cycle_iters': 10000,
    
    'weight_decay': 1e-2,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    
    'it_warm': 5,
    'it_n': 100,
}

def main(configs):
    model = BasicsTransformerLM(
        vocab_size=configs['vocab_size']
        context_length=configs['context_length'],
        d_model=configs['d_model'],
        num_layers=configs['num_layers'],
        num_heads=configs['num_heads'],
        d_ff=configs['d_ff'],
        rope_theta=configs['rope_theta'],
    )
    opt = AdamW(
        params=model.parameters(),
        lr=configs['max_learning_rate'],
        betas=configs['betas'],
        eps=configs['eps'],
        weight_decay=configs['weight_decay'],
    )
    
    model.train()
    for step in range(configs['it_warm'] + configs['it_n']):
        
    

if __name__ == "__main__":
    sys.exit(main())