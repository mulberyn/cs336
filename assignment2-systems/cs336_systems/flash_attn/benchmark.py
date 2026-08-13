"""
benchmark_flash_attention.py

对比：
  1. Triton FlashAttention-2（前向 triton kernel + 反向 triton kernel）
  2. 纯 PyTorch 实现（不使用 flash attention，即标准 attention: softmax(QK^T/sqrt(d))V ）

在单张 B200 上，batch_size=1，causal=True，
对 (seq_len, d_model, dtype) 做笛卡尔积扫描，报告 forward / backward / end-to-end 延迟。
"""

import csv
import itertools
import math
import os
import time
import torch
import pandas as pd
from triton.testing import do_bench
from triton.runtime.errors import OutOfResources

# ------------------------------------------------------------------
# 1. 待比较的两种实现
# ------------------------------------------------------------------

from flash_attn_triton import FlashAttentionTriton  # noqa: E402


def pytorch_attention(Q, K, V, is_causal=True):
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if is_causal:
        Nq, Nk = Q.shape[-2], K.shape[-2]
        mask = torch.triu(
            torch.ones(Nq, Nk, dtype=torch.bool, device=Q.device), diagonal=1
        )
        S = S.masked_fill(mask, torch.finfo(S.dtype).min)

    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    return O


# ------------------------------------------------------------------
# 2. 扫描配置
# ------------------------------------------------------------------

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_MODELS = [16, 32, 64, 128]
DTYPES = [torch.bfloat16, torch.float32]

BATCH_SIZE = 1
IS_CAUSAL = True
DEVICE = "cuda"

WARMUP_ITERS = 25
BENCH_ITERS = 100

CSV_PATH = "results/flash_attention_benchmark_results.csv"

CSV_FIELDS = [
    "seq_len", "d_model", "dtype",
    "triton_fwd_ms", "triton_bwd_ms", "triton_e2e_ms",
    "torch_fwd_ms", "torch_bwd_ms", "torch_e2e_ms",
    "speedup_fwd", "speedup_bwd", "speedup_e2e",
    "status",
]


# ------------------------------------------------------------------
# 3. CSV 增量写入工具
# ------------------------------------------------------------------

def init_csv(path):
    """如果文件不存在，写入表头；如果已存在，不覆盖（方便中断后续跑）。"""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_row_to_csv(path, row: dict):
    """每跑完一个配置就立刻追加写入一行，并 flush，防止中途崩溃丢数据。"""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_finished_configs(path):
    """
    读取已有 CSV，返回已经跑过的 (seq_len, d_model, dtype) 集合，
    用于脚本中断后重新运行时跳过已完成的配置（断点续跑）。
    """
    finished = set()
    if not os.path.exists(path):
        return finished
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            finished.add((int(row["seq_len"]), int(row["d_model"]), str(row["dtype"])))
    except Exception:
        pass
    return finished


# ------------------------------------------------------------------
# 4. 单次配置的 benchmark
# ------------------------------------------------------------------

def make_inputs(seq_len, d_model, dtype):
    torch.manual_seed(0)
    shape = (BATCH_SIZE, seq_len, d_model)
    Q = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
    K = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
    V = torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True)
    return Q, K, V


def bench_one(fn_forward_only, fn_forward_for_backward, Q, K, V):
    def fwd():
        with torch.no_grad():
            return fn_forward_only()

    fwd_ms = do_bench(fwd, warmup=WARMUP_ITERS, rep=BENCH_ITERS)

    O_cache = {}

    def setup():
        Q.grad = None
        K.grad = None
        V.grad = None
        O = fn_forward_for_backward()
        O_cache["O"] = O
        O_cache["grad"] = torch.randn_like(O)

    def bwd():
        O_cache["O"].backward(O_cache["grad"], retain_graph=False)

    bwd_ms = _bench_with_setup(setup, bwd)

    def e2e():
        Q.grad = None
        K.grad = None
        V.grad = None
        O = fn_forward_for_backward()
        grad = torch.randn_like(O)
        O.backward(grad)

    e2e_ms = do_bench(e2e, warmup=WARMUP_ITERS, rep=BENCH_ITERS)

    return fwd_ms, bwd_ms, e2e_ms


def _bench_with_setup(setup_fn, measured_fn, warmup=WARMUP_ITERS, rep=BENCH_ITERS):
    for _ in range(warmup):
        setup_fn()
        measured_fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(rep):
        setup_fn()
        torch.cuda.synchronize()
        start.record()
        measured_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    mid = len(times) // 2
    return times[mid]


def make_result_row(seq_len, d_model, dtype_str, status, **metrics):
    row = {
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype": dtype_str,
        "triton_fwd_ms": "", "triton_bwd_ms": "", "triton_e2e_ms": "",
        "torch_fwd_ms": "", "torch_bwd_ms": "", "torch_e2e_ms": "",
        "speedup_fwd": "", "speedup_bwd": "", "speedup_e2e": "",
        "status": status,
    }
    row.update(metrics)
    return row


# ------------------------------------------------------------------
# 5. 主循环：扫描所有配置，边跑边存 CSV
# ------------------------------------------------------------------

def run_benchmark(resume=True):
    init_csv(CSV_PATH)
    finished = load_finished_configs(CSV_PATH) if resume else set()

    for seq_len, d_model, dtype in itertools.product(SEQ_LENS, D_MODELS, DTYPES):
        dtype_str = str(dtype).replace("torch.", "")
        cfg_key = (seq_len, d_model, dtype_str)
        cfg_name = f"seq={seq_len}, d={d_model}, dtype={dtype_str}"

        if cfg_key in finished:
            print(f"Skipping {cfg_name} (already in CSV).")
            continue

        print(f"Running {cfg_name} ...")
        t0 = time.time()

        try:
            Q, K, V = make_inputs(seq_len, d_model, dtype)

            # ---------------- Triton FlashAttention-2 ----------------
            def triton_fwd():
                return FlashAttentionTriton.apply(Q, K, V, IS_CAUSAL)

            triton_fwd_ms, triton_bwd_ms, triton_e2e_ms = bench_one(
                triton_fwd, triton_fwd, Q, K, V
            )

            # ---------------- 纯 PyTorch attention ----------------
            def torch_fwd():
                return pytorch_attention(Q, K, V, IS_CAUSAL)

            torch_fwd_ms, torch_bwd_ms, torch_e2e_ms = bench_one(
                torch_fwd, torch_fwd, Q, K, V
            )

            row = make_result_row(
                seq_len, d_model, dtype_str, status="ok",
                triton_fwd_ms=triton_fwd_ms,
                triton_bwd_ms=triton_bwd_ms,
                triton_e2e_ms=triton_e2e_ms,
                torch_fwd_ms=torch_fwd_ms,
                torch_bwd_ms=torch_bwd_ms,
                torch_e2e_ms=torch_e2e_ms,
                speedup_fwd=torch_fwd_ms / triton_fwd_ms,
                speedup_bwd=torch_bwd_ms / triton_bwd_ms,
                speedup_e2e=torch_e2e_ms / triton_e2e_ms,
            )

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at {cfg_name}, skipping.")
            row = make_result_row(seq_len, d_model, dtype_str, status="OOM")

        except OutOfResources as e:
            print(f"  Triton OutOfResources at {cfg_name}: {e}")
            row = make_result_row(seq_len, d_model, dtype_str, status="OutOfResources")

        except Exception as e:
            print(f"  Unexpected error at {cfg_name}: {type(e).__name__}: {e}")
            row = make_result_row(seq_len, d_model, dtype_str, status=f"error:{type(e).__name__}")

        finally:
            torch.cuda.empty_cache()

        # 每个配置跑完立刻落盘，避免脚本中途崩溃丢失前面的结果
        append_row_to_csv(CSV_PATH, row)
        print(f"  done in {time.time() - t0:.1f}s, saved to {CSV_PATH}")

    return pd.read_csv(CSV_PATH)


if __name__ == "__main__":
    df = run_benchmark(resume=True)

    print(df.to_markdown(index=False, floatfmt=".3f"))