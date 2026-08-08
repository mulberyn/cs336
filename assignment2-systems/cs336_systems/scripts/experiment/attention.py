import argparse
import time
import torch
from itertools import product

# Suppress TF32 warning and enable fastest matmul
torch.set_float32_matmul_precision('high')


def attention_fn(q, k, v):
    """Standard scaled dot‑product attention (single head)."""
    scale = q.size(-1) ** 0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Use torch.compile on the attention function.'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Data type for inputs (default: float32).'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    use_compile = args.compile

    device = torch.device('cuda')
    batch_size = 8
    d_models = [16, 32, 64, 128]          # as required
    seq_lens = [256, 1024, 4096, 8192, 16384]
    num_iter = 100
    warmup = 10

    # Build the function to benchmark
    if use_compile:
        fn = torch.compile(attention_fn, mode="default")
        mode_str = "Compiled"
    else:
        fn = attention_fn
        mode_str = "Eager"

    print(f"Mode: {mode_str}, dtype: {args.dtype}")
    print(f"{'Seq':<6} | {'d':<4} | {'Fwd Time (ms)':<15} | {'Bwd Time (ms)':<15} | {'Memory (MiB)':<12} | Status")
    print("-" * 80)

    for L, d in product(seq_lens, d_models):
        status = "OK"
        fwd_time = bwd_time = mem_alloc = None
        try:
            # Create inputs with the chosen dtype
            Q = torch.randn(batch_size, L, d, device=device, dtype=dtype, requires_grad=True)
            K = torch.randn(batch_size, L, d, device=device, dtype=dtype, requires_grad=True)
            V = torch.randn(batch_size, L, d, device=device, dtype=dtype, requires_grad=True)

            # ---- Warmup ----
            # For compiled mode, a few warmup iterations trigger the actual compilation.
            # The first calls may be slow but are not timed.
            for _ in range(warmup):
                out = fn(Q, K, V)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()
                # clear gradients
                Q.grad = None
                K.grad = None
                V.grad = None

            # ---- Measure forward time ----
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_iter):
                out = fn(Q, K, V)
                torch.cuda.synchronize()
            fwd_time = (time.perf_counter() - start) / num_iter * 1000  # ms

            # ---- Measure memory before backward ----
            torch.cuda.synchronize()
            mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MiB

            # ---- Measure backward time ----
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_iter):
                out = fn(Q, K, V)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()
                Q.grad = None
                K.grad = None
                V.grad = None
            bwd_time = (time.perf_counter() - start) / num_iter * 1000  # ms

        except RuntimeError as e:
            if "out of memory" in str(e):
                status = "OOM"
                torch.cuda.empty_cache()
            else:
                raise e

        # Format output
        fwd_str = f"{fwd_time:.3f}" if fwd_time is not None else "OOM"
        bwd_str = f"{bwd_time:.3f}" if bwd_time is not None else "OOM"
        mem_str = f"{mem_alloc:.2f}" if mem_alloc is not None else "OOM"
        print(f"{L:<6} | {d:<4} | {fwd_str:<15} | {bwd_str:<15} | {mem_str:<12} | {status}")


if __name__ == "__main__":
    main()