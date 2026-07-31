import argparse
import wandb
import sys
import torch
import numpy as np
import random
import math
import os
from tqdm import tqdm

from cs336_basics.modules import TransformerLM
from cs336_basics.trainer import *

DEFAULT_CONFIG = {
    # 模型配置
    'vocab_size': 10000,
    'context_length': 256,
    'd_model': 512,
    'num_layers': 4,
    'd_ff': 1344,
    'num_heads': 16,
    'rope_theta': 10000.0,
    
    # 优化器配置
    'lr_max': 1e-3,
    'lr_min': 1e-4,
    't_warm': 500,
    't_end': 10000,
    'weight_decay': 1e-2,
    'beta1': 0.9,
    'beta2': 0.95,
    'eps': 1e-8,
    'max_l2_norm': 1.0,
    
    # 训练配置
    'batch_size': 32,
    'train_steps': 6000,
    'val_interval': 100,
    'val_batch': 10,
    'save_intervals': 1000,
    'log_intervals': 1,
    'save_ckp_path': './checkpoints',
    'resume_ckp': None,
    'seed': random.randint(0, int(1e9)), 
    
    # 数据配置
    'train_data_path': './data/TinyStoriesV2-GPT4-train.bin',
    'valid_data_path': './data/TinyStoriesV2-GPT4-valid.bin',
    'device': 'auto',
    
    # 记录配置
    'wandb_project': 'cs336-transformer',
    'wandb_run_name': None,
    'no_wandb': False,
    'export_final': './model_final.pt',   # 最终模型导出路径
}

def load_prase():
    parser = argparse.ArgumentParser(description='训练参数')
    
    model_group = parser.add_argument_group('模型配置', '训练模型的参数')
    optimizer_group = parser.add_argument_group('优化器配置', '模型采用的优化器参数')
    training_group = parser.add_argument_group('训练配置', '训练过程时的参数')
    data_group = parser.add_argument_group('数据配置', '训练数据的参数')
    record_group = parser.add_argument_group('记录配置', '导出及wandb参数')
    
    # 模型配置
    model_group.add_argument('--vocab_size', type=int, default=DEFAULT_CONFIG['vocab_size'], help='Size of vocabulary')
    model_group.add_argument('--context_length', type=int, default=DEFAULT_CONFIG['context_length'], help='Maximum sequence length')
    model_group.add_argument('--d_model', type=int, default=DEFAULT_CONFIG['d_model'], help='Model dimension')
    model_group.add_argument('--num_layers', type=int, default=DEFAULT_CONFIG['num_layers'], help='Number of transformer layers')
    model_group.add_argument('--d_ff', type=int, default=DEFAULT_CONFIG['d_ff'], help='FFN dimension')
    model_group.add_argument('--num_heads', type=int, default=DEFAULT_CONFIG['num_heads'], help='Number of attention heads')
    model_group.add_argument('--rope_theta', type=float, default=DEFAULT_CONFIG['rope_theta'], help='RoPE theta parameter')
    
    # 优化器配置
    optimizer_group.add_argument('--lr_max', type=float, default=DEFAULT_CONFIG['lr_max'], help='Maximum learning rate')
    optimizer_group.add_argument('--lr_min', type=float, default=DEFAULT_CONFIG['lr_min'], help='Minimum learning rate')
    optimizer_group.add_argument('--t_warm', type=int, default=DEFAULT_CONFIG['t_warm'], help='Warmup iterations')
    optimizer_group.add_argument('--t_end', type=int, default=DEFAULT_CONFIG['t_end'], help='Cosine annealing iterations')
    
    optimizer_group.add_argument('--weight_decay', type=float, default=DEFAULT_CONFIG['weight_decay'], help='Weight decay')
    optimizer_group.add_argument('--beta1', type=float, default=DEFAULT_CONFIG['beta1'], help='Adam beta1')
    optimizer_group.add_argument('--beta2', type=float, default=DEFAULT_CONFIG['beta2'], help='Adam beta2')
    optimizer_group.add_argument('--eps', type=float, default=DEFAULT_CONFIG['eps'], help='Adam epsilon')
    
    optimizer_group.add_argument('--max_l2_norm', type=float, default=DEFAULT_CONFIG['max_l2_norm'], help='Max l2 norm of Gradient clipping norm')
    
    # 训练配置
    training_group.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'], help='Batch size')
    training_group.add_argument('--train_steps', type=int, default=DEFAULT_CONFIG['train_steps'], help='Total training steps')
    training_group.add_argument('--val_interval', type=int, default=DEFAULT_CONFIG['val_interval'], help='Validation interval')
    training_group.add_argument('--val_batch', type=int, default=DEFAULT_CONFIG['val_batch'], help='Number of validation batches')
    training_group.add_argument('--save_intervals', type=int, default=DEFAULT_CONFIG['save_intervals'], help='Checkpoint save interval')
    training_group.add_argument('--log_intervals', type=int, default=DEFAULT_CONFIG['log_intervals'], help='Logging interval')
    training_group.add_argument('--save_ckp_path', type=str, default=DEFAULT_CONFIG['save_ckp_path'], help='Checkpoint save directory')
    training_group.add_argument('--resume_ckp', type=str, default=DEFAULT_CONFIG['resume_ckp'], help='Path to checkpoint to resume from')
    training_group.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'], help='随机种子')
    
    # 数据配置
    data_group.add_argument('--train_data_path', type=str, default='./data/TinyStoriesV2-GPT4-train.bin', help='Path to training data')
    data_group.add_argument('--valid_data_path', type=str, default='./data/TinyStoriesV2-GPT4-valid.bin', help='Path to validation data')
    data_group.add_argument('--device', type=str, default=DEFAULT_CONFIG['device'], help='Device: auto, cpu, cuda, mps')
    
    # 记录配置（含导出路径）
    record_group.add_argument('--wandb_project', type=str, default=DEFAULT_CONFIG['wandb_project'], help='Wandb project name')
    record_group.add_argument('--wandb_run_name', type=str, default=DEFAULT_CONFIG['wandb_run_name'], help='Wandb run name')
    record_group.add_argument('--no_wandb', action='store_true', default=DEFAULT_CONFIG['no_wandb'], help='Disable wandb logging')
    record_group.add_argument('--export_final', type=str, default=DEFAULT_CONFIG['export_final'], help='Path to save final model for inference')
    
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataset, batch_size, context_length, device, num_batches):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = data_loading(dataset, batch_size, context_length, device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main():
    # ========== 载入命令行参数 ==========
    args = load_prase()
    
    # ========== 设置随机种子 ==========
    set_seed(args.seed)
    
    # ========== 设备设定 ==========
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # ========== 加载数据 ==========
    train_data = np.memmap(args.train_data_path, dtype=np.uint32)
    val_data = np.memmap(args.valid_data_path, dtype=np.uint32)
    assert len(train_data) > args.context_length + 1, "训练数据太短"
    assert len(val_data) > args.context_length + 1, "验证数据太短"
    
    # ========== 构建模型 ==========
    if args.d_ff is None:
        args.d_ff = int(8 / 3 * args.d_model + 63) // 64 * 64
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    ).to(device)
    
    # ========== 构建优化器 ==========
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    
    # ========== 初始化 wandb 并记录参数类型 ==========
    if args.no_wandb:
        wandb.init(mode='disabled')
    else:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
        print(f"W&B 初始化成功: {wandb.run.name}")
    
    # ========== 恢复 checkpoint ==========
    start_step = 0
    if args.resume_ckp is not None:
        start_step = load_checkpoint(args.resume_ckp, model, optimizer)
        print(f"从 {args.resume_ckp} 恢复，从第 {start_step} 步继续")
        
    # ========== 初始 loss ==========
    init_val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.val_batch)
    print(f"初始验证损失: {init_val_loss:.4f}")
    if not args.no_wandb:
        wandb.log({"val_loss": init_val_loss, "val_perplexity": math.exp(init_val_loss), "step": 0})
        
    # ========== 开始训练 ==========
    pbar = tqdm(range(start_step, args.train_steps), 
                desc="Training", 
                initial=start_step, 
                total=args.train_steps)
    for step in pbar:
        # ========== 获取当前批次 ==========
        inputs, targets = data_loading(train_data, args.batch_size, args.context_length, device)
        
        # ========== 前向传播（输出 logits，计算 loss） ==========
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        
        # ========== 反向传播（梯度清零、优化器反向传播） ==========
        optimizer.zero_grad()
        loss.backward()
        
        # ========== 梯度裁剪（并获取范数） ==========
        grad_norm = gradient_clipping(model.parameters(), args.max_l2_norm)
        
        # ========== 优化器更新 ==========
        optimizer.step()
        
        # ========== 更新学习率 ==========
        lr = get_lr_cosine_schedule(
            step + 1, args.lr_max, args.lr_min, args.t_warm, args.t_end
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # ========== 日志记录（每 log_intervals 步） ==========
        if (step + 1) % args.log_intervals == 0:
            perplexity = math.exp(loss.item())
            # 计算吞吐量（需要记录时间）
            # tokens_per_sec = (batch_size * context_length) / elapsed_time
            # 这里假设你已经计算了 elapsed_time
            print(f"Step {step + 1}/{args.train_steps} | loss={loss.item():.4f} | ppl={perplexity:.2f} | lr={lr:.2e} | grad_norm={grad_norm:.2f}")
            if not args.no_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "perplexity": perplexity,
                    "learning_rate": lr,
                    "grad_norm": grad_norm,
                    "step": step + 1,
                })
        
        # ========== 验证（每 val_interval 步） ==========
        if (step + 1) % args.val_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.val_batch)
            val_ppl = math.exp(val_loss)
            print(f"Step {step+1} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            if not args.no_wandb:
                wandb.log({
                    "val_loss": val_loss,
                    "val_perplexity": val_ppl,
                    "step": step + 1,
                })
        
        # ========== 验证（每 val_interval 步） ==========
        if (step + 1) % args.save_intervals == 0:
            ckpt_path = os.path.join(args.save_ckp_path, f"checkpoint_{step+1:08d}.pt")
            os.makedirs(args.save_ckp_path, exist_ok=True)
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
        
    final_loss = loss.item()   # 最后一次训练的 loss
    final_step = args.train_steps
    export_path = args.export_final
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    # 保存前将模型移至 CPU，避免设备绑定
    model.cpu()
    export_dict = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
        },
        "final_loss": final_loss,
        "final_step": final_step,
    }
    torch.save(export_dict, export_path)
    print(f"最终模型已保存至 {export_path}")

    # ========== 12. 将最终模型上传到 W&B（作为 artifact） ==========
    if not args.no_wandb:
        artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
        artifact.add_file(export_path)
        wandb.log_artifact(artifact)
        wandb.log({"final_loss": final_loss, "final_step": final_step})

    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())