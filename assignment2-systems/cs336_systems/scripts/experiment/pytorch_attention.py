import torch
import time
from itertools import product

def run_benchmark():
    device = torch.device('cuda')
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    num_iter = 100
    warmup = 10
    
    print(f"{'Seq Len':<8} | {'d_model':<6} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Memory (MiB)':<12} | Status")
    print("-" * 70)
    
    for L, d in product(seq_lens, d_models):
        status = "OK"
        fwd_time = bwd_time = mem_alloc = None
        try:
            # 随机 Q, K, V，开启梯度
            Q = torch.randn(batch_size, L, d, device=device, requires_grad=True)
            K = torch.randn(batch_size, L, d, device=device, requires_grad=True)
            V = torch.randn(batch_size, L, d, device=device, requires_grad=True)

            # 预热
            for _ in range(warmup):
                out = torch.matmul(torch.softmax(torch.matmul(Q, K.transpose(1, 2)), dim=-1), V)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()
                Q.grad = None; K.grad = None; V.grad = None

            # 计时100次前向传播
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_iter):
                out = torch.matmul(torch.softmax(torch.matmul(Q, K.transpose(1, 2)), dim=-1), V)
                torch.cuda.synchronize()
            fwd_time = (time.perf_counter() - start) / num_iter * 1000  # ms

            # 测量反向传播开始前的内存使用量
            torch.cuda.synchronize()
            mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MiB

            # 计时100次反向传播
            start = time.perf_counter()
            for _ in range(num_iter):
                out = torch.matmul(torch.softmax(torch.matmul(Q, K.transpose(1, 2)), dim=-1), V)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()
                Q.grad = None; K.grad = None; V.grad = None
            bwd_time = (time.perf_counter() - start) / num_iter * 1000  # ms

        except RuntimeError as e:
            if "out of memory" in str(e):
                status = "OOM"
                torch.cuda.empty_cache()
            else:
                raise e
        
        # 打印结果（或OOM）
        if status == "OOM":
            print(f"{L:<8} | {d:<6} | {'OOM':<10} | {'OOM':<10} | {'OOM':<12} | OOM")
        else:
            print(f"{L:<8} | {d:<6} | {fwd_time:<10.3f} | {bwd_time:<10.3f} | {mem_alloc:<12.2f} | OK")

if __name__ == "__main__":
    run_benchmark()