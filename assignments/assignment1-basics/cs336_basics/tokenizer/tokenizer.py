import heapq
from .train import pretokenize
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[str, int], 
        merges: list[tuple[str, str]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.pair_to_priority = {pair: i for i, pair in enumerate(merges)}
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
        """Encode a string into a list of token IDs using the BPE tokenizer.

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

            byte_seq = token.encode("utf-8")
            tokens = [bytes([b]) for b in byte_seq]

            while True:
                best_priority = None
                best_pos = -1
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.pair_to_priority:
                        priority = self.pair_to_priority[pair]
                        if best_priority is None or priority < best_priority:
                            best_priority = priority
                            best_pos = i
                if best_pos == -1:
                    break
                new_token = tokens[best_pos] + tokens[best_pos + 1]
                tokens = tokens[:best_pos] + [new_token] + tokens[best_pos + 2:]
            
            for tok in tokens:
                encoded_ids.append(self.vocab_reverse[tok])

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
        all_bytes = b''.join(self.vocab[id] for id in ids)
        return all_bytes.decode("utf-8", errors="replace")