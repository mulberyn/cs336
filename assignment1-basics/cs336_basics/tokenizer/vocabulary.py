from cs336_basics.tokenizer.ctypes import Vocabulary


def initialize_vocab(
    special_tokens: list[str],
) -> Vocabulary:
    vocab = {}
    idx = 0
    
    for byte_value in range(256):
        vocab[idx] = bytes([byte_value])
        idx += 1
    for token in special_tokens:
        vocab[idx] = token.encode("utf-8")
        idx += 1
        
    return vocab