import regex as re

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(
    text: str, 
    special_tokens: list[str] = []
) -> list[str]:
    """GPT‑2 风格预分词，支持特殊 token 整体保留"""
    if not special_tokens:
        return re.findall(GPT2_PATTERN, text)
    
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    parts = re.split(f"({pattern})", text)
    
    tokens = []
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            tokens.append(part)
        else:
            tokens.extend(re.findall(GPT2_PATTERN, part))
    return tokens