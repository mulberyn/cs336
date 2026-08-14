import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def benchmark_rank(rank, world_size, data_size_mb):
    """
    每个进程执行的基准测试函数（Gloo CPU 版本）。
    """
    # 初始化 Gloo 进程组（单节点，固定端口）
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    # 创建 CPU 张量（float32）
    num_elements = data_size_mb * 1024 * 1024 // 4   # 每 MB 对应 262144 个 float32
    tensor = torch.randn(num_elements, dtype=torch.float32)

    # ---- warmup ----
    warmup_iters = 5
    for _ in range(warmup_iters):
        dist.all_reduce(tensor)   # Gloo 是阻塞操作，返回即完成

    # ---- 测量 ----
    measure_iters = 10
    times = []   # 当前进程每次测量的耗时（毫秒）
    for _ in range(measure_iters):
        start = time.perf_counter()
        dist.all_reduce(tensor)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0
        times.append(elapsed_ms)

    # 收集所有等级的测量结果
    all_times = [None] * world_size
    dist.all_gather_object(all_times, times)

    # 仅 rank0 打印汇总统计
    if rank == 0:
        flat = [t for sublist in all_times for t in sublist]
        avg = sum(flat) / len(flat)
        std = (sum((x - avg) ** 2 for x in flat) / len(flat)) ** 0.5
        print(f"World size: {world_size:2d}, Data size: {data_size_mb:4d} MB, "
              f"Avg time: {avg:8.2f} ms, Std: {std:6.2f} ms")

    dist.destroy_process_group()


def main():
    # 根据 CPU 核心数适当限制最大进程数（这里用 6 没问题）
    world_sizes = [2, 4, 6]
    data_sizes_mb = [1, 10, 100, 1000]   # 1 MB ~ 1 GB

    for world_size in world_sizes:
        for data_size in data_sizes_mb:
            mp.spawn(benchmark_rank,
                     args=(world_size, data_size),
                     nprocs=world_size,
                     join=True)
            # 等待进程组完全释放，避免端口冲突
            time.sleep(1)

    print("\n所有基准测试完成。请根据打印表格分析通信时间与数据量、进程数的关系。")


if __name__ == "__main__":
    main()