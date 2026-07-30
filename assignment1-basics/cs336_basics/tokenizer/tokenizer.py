import heapq
from .train import pretokenize
from typing import Iterable


class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        """
        vocab: token_id(int) -> token(bytes), bytes is converted from str by utf-8
        vocab_reverse: token(bytes) -> token_id
        merges: (bytes, bytes)
        """
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        # 从合并的 pair 到合并 id 的映射, id 越小合并优先级越高
        self.pair_to_priority = {pair: i for i, pair in enumerate(merges)}
        # 从长的开始 special_token 开始进行匹配
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else []
    
    
    @classmethod
    def from_file(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        """Load a tokenizer from vocabulary and merges files.

        Args:
            vocab_filepath: Path to the vocabulary file.
            merges_filepath: Path to the merges file.
            special_tokens: Optional list of special tokens to include in the tokenizer.
        """
        vocab = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as vocab_file:
            for line in vocab_file:
                token, token_id = line.strip().split()
                vocab[token] = int(token_id)
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as merges_file:
            for line in merges_file:
                if line.startswith('#'):
                    continue  # Skip comment lines
                merge = tuple(line.strip().split())
                merges.append(merge)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    
    def encode(
        self, 
        text: str,
    ) -> list[int]:
        """
        Encode a string into a list of token IDs using the BPE tokenizer.

        Args:
            text: The input string to encode.

        Returns:
            A list of integer token IDs corresponding to the input string.
        """
        if not text:
            return []
        encoded_ids = []
        pre_tokens = pretokenize(text, self.special_tokens)
        
        for token in pre_tokens:
            if token in self.special_tokens: 
                encoded_ids.append(self.vocab_reverse[token.encode("utf-8")])
                continue
            byte_seq = token.encode("utf-8") # 将 str 转换成 bytes(字节级)
            tokens = [bytes([b]) for b in byte_seq] # 转换成字节序列
            
            while True:
                best_priority = None
                best_pos = -1
                
                # 寻找合并优先级最高的位置
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.pair_to_priority:
                        priority = self.pair_to_priority[pair]
                        if best_priority is None or priority < best_priority:
                            best_priority = priority
                            best_pos = i
                
                if best_pos == -1:
                    break
                
                # 进行合并
                new_token = tokens[best_pos] + tokens[best_pos + 1] 
                tokens = tokens[:best_pos] + [new_token] + tokens[best_pos + 2:]
            
            # 合并结束后，将所有的 token 对应的 token_id 记录下来
            for token in tokens:
                encoded_ids.append(self.vocab_reverse[token])

        return encoded_ids
    
    
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    
    def decode(
        self, 
        ids: list[int],
    ) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            token_ids: A list of integer token IDs to decode.

        Returns:
            The decoded string corresponding to the input token IDs.
        """
        all_bytes = b''.join(self.vocab[id] for id in ids) # 将 token_id 序列转换成 bytes
        return all_bytes.decode("utf-8", errors="replace") # 从 bytes 转换成 Unicode 文本