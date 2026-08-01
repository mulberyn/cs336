import torch
import argparse
import sys

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.modules import TransformerLM, softmax

DEFAULT_CONFIG = {
    'model_path': './out/model/model_final.pt',
    'tokenizer_dir': './out/tokenizer',
    'prompt': 'Once upon a time,',
    'max_new_tokens': 100,
    'temperature': 0.8,
    'top_p': 0.9,
    'stop_token': '<|endoftext|>',
    'device': 'auto',
}

def load_parse():
    parser = argparse.ArgumentParser(description="从训练好的 TransformerLM 生成文本")
    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG['model_path'], help="训练好的模型文件 (.pt)")
    parser.add_argument("--tokenizer_dir", type=str, default=DEFAULT_CONFIG['tokenizer_dir'], help="词表文件 (vocab.json)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_CONFIG['prompt'], help="生成的前缀文本")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_CONFIG['max_new_tokens'], help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG['temperature'], help="温度 (0 表示贪婪采样)")
    parser.add_argument("--top_p", type=float, default=DEFAULT_CONFIG['top_p'], help="Top‑p 值 (1.0 表示不使用 top‑p)")
    parser.add_argument("--stop_token", type=str, default=DEFAULT_CONFIG['stop_token'], help="停止生成的特殊 token")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG['device'], choices=["auto", "cpu", "cuda", "mps"], help="运行设备 (auto 自动选择)")
    return parser.parse_args()


def temperature_softmax(logits, temperature=1.0):
    # logits: shape (vocab_size,)
    if temperature == 0:
        # 退化为 one-hot（取 argmax）
        # 只有 logits 最大的概率为 1（只取最大的）
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
    else:
        probs = softmax(logits / temperature, dim=-1)
    return probs


def sample_top_p(probs, p):
    # sorted_probs 为排序后的概率, sorted_indices 为排序后的索引
    sorted_probs, sorted_indices = torch.sort(probs, descending=True) # 均为(vocab_size,)
    # 对 sorted_probs 做一个前缀和
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # mask 记录 cumsum 哪些位置 <= p
    mask = cumsum <= p
    # 强制选概率最大的
    mask[..., 0] = True
    # 将 mask 为 False 的位置都变成 0
    sorted_probs = sorted_probs * mask
    # 重新归一化成概率和为1
    sorted_probs = sorted_probs / sorted_probs.sum()
    # 按照新的概率分布，随机抽一个下标
    idx = torch.multinomial(sorted_probs, 1) # idx(1,)
    # 找到原本的索引
    next_id = sorted_indices[idx] # next_id(1,)
    return next_id.item() #获取其数字



def generate(
    model,
    tokenizer,
    prompt: str, 
    max_new_tokens: int, 
    temperature: float, 
    top_p: float,
    stop_token: str,
    device,
) -> str:
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.Tensor([input_ids], device=device) # input_tensor (1, seq_len)
    eos_token_id = tokenizer.encode(stop_token)
    model.eval()
    
    for _ in range(max_new_tokens):
        logits = model(input_tensor) # logits (1, seq_len, vocab_size)
        # 取出最后一个单词（也就是预测的下一个词）的全部概率
        next_logits = logits[0, -1, :] 
        # 引入温度
        probs = temperature_softmax(next_logits, temperature)
        # 引入 top_p
        if top_p < 1.0:
            next_id = sample_top_p(probs, top_p)
        else:
            next_id = torch.multinomial(probs, 1)
        # 如果遇到结束标记，那么截止
        if next_id == eos_token_id:
            break
        # 更新全部文本
        input_ids.append(next_id)
        # 更新 input_tensor
        next_tensor = torch.tensor([[next_id]], device=device)
        input_tensor = torch.cat([input_tensor, next_tensor], dim=1)
    
    return tokenizer.decode(input_ids)

def main():
    args = load_parse()

    # 确定设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"使用设备: {device}")

    # 加载 tokenizer
    print("加载 tokenizer ...")
    tokenizer = Tokenizer.from_files(args.tokenzier_dir + '/vocab.json', args.tokenzier_dir + '/merges.json',
                                     special_tokens=[args.stop_token])

    # 加载模型
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    state_dict = checkpoint["model_state_dict"]

    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        device=device,
        dtype=None,          # 默认 float32
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # 4. 生成
    print(f"生成文本 (prompt: '{args.prompt}') ...")
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_token=args.stop_token,
        device=device,
        context_length=model_config["context_length"],
    )

    print("\n" + "=" * 50)
    print("生成结果:")
    print(output)
    print("=" * 50)


if __name__ == "__main__":
    sys.exit(main())