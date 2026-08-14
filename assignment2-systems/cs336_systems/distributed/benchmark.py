import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda import Event

def benchmark_rank(rank, world_size, data_size_mb, backend='nccl'):
    """
    每个进程执行的基准测试函数。
    初始化进程组，分配设备，执行 warmup，测量 all-reduce 时间，
    并通过 all_gather_object 收集所有等级的测量结果。
    """
    # 初始化进程组（单节点，使用固定端口）
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # 分配 GPU 设备（假设每个进程独占一张 GPU）
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 创建浮点张量，数据量按 MB 计算
    num_elements = data_size_mb * 1024 * 1024 // 4  # float32 占 4 字节
    tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
    
    # ---- warmup ----
    warmup_iters = 5
    for _ in range(warmup_iters):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()   # 确保操作完成
    
    # ---- 测量 ----
    measure_iters = 10
    times = []   # 当前进程每次测量的耗时（毫秒）
    for _ in range(measure_iters):
        start = Event(enable_timing=True)
        end = Event(enable_timing=True)
        start.record()
        dist.all_reduce(tensor)
        end.record()
        torch.cuda.synchronize()   # 等待 GPU 完成
        elapsed = start.elapsed_time(end)  # 毫秒
        times.append(elapsed)
    
    # 收集所有等级的测量结果（列表的列表）
    all_times = [None] * world_size
    dist.all_gather_object(all_times, times)
    
    # 仅 rank0 打印汇总统计
    if rank == 0:
        # 展平所有等级的测量值
        flat = [t for sublist in all_times for t in sublist]
        avg = sum(flat) / len(flat)
        std = (sum((x - avg) ** 2 for x in flat) / len(flat)) ** 0.5
        print(f"World size: {world_size:2d}, Data size: {data_size_mb:4d} MB, "
              f"Avg time: {avg:7.2f} ms, Std: {std:6.2f} ms")
    
    dist.destroy_process_group()


def main():
    # 根据实际可用 GPU 数量调整可测试的世界大小
    num_gpus = torch.cuda.device_count()
    desired_world_sizes = [2, 4, 6]
    world_sizes = [w for w in desired_world_sizes if w <= num_gpus]
    if not world_sizes:
        print("错误：至少需要 2 张 GPU 才能进行测试。")
        return
    if len(world_sizes) < len(desired_world_sizes):
        print(f"警告：仅检测到 {num_gpus} 张 GPU，将测试 world sizes: {world_sizes}")
    
    data_sizes_mb = [1, 10, 100, 1000]   # 1 MB ~ 1 GB
    
    backend = 'nccl'   # 单节点多 GPU 推荐使用 NCCL
    
    for world_size in world_sizes:
        for data_size in data_sizes_mb:
            # spawn 进程，每个进程运行 benchmark_rank
            mp.spawn(benchmark_rank,
                     args=(world_size, data_size, backend),
                     nprocs=world_size,
                     join=True)
            # 让进程组完全释放，避免端口占用冲突
            time.sleep(1)
    
    print("\n所有基准测试完成。根据打印表格分析通信开销与数据规模、GPU数量的关系。")


if __name__ == "__main__":
    main()