import regex as re

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(
    text: str,
    special_tokens: list[str] = [],
) -> list[str]:
    if not special_tokens:
        return re.findall(GPT2_PATTERN, text)
    
    special_tokens_pattern = "|".join(
        re.escape(token) 
        for token in special_tokens
    )
    parts = re.split(
        f"({special_tokens_pattern})",
        text,
    )
    
    token_sequence = []
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            token_sequence.append(part)
        else:
            token_sequence.extend(re.findall(GPT2_PATTERN, part))
            
    return token_sequence


if __name__ == "__main__":
    text = (
        "Hello world<|endoftext|>"
        "How are you?"
    )
    tokens = pretokenize(text, special_tokens=["<|endoftext|>"])
    print(tokens)