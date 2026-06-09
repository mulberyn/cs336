def initialize_vocab(
    special_tokens: list[str]
) -> dict[int, bytes]:
    """构建初始词汇表：0‑255 为单字节，接着是特殊 token"""
    
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    return vocab