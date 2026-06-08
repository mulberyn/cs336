from cs336_basics.tokenizer.corpus import load_text
from cs336_basics.tokenizer.pretokenize import pretokenize
from cs336_basics.tokenizer.vocabulary import (
  initialize_vocab,
)
from cs336_basics.tokenizer.merges import (
    count_pairs,
    find_best_pair,
    apply_merge,
)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text: str = load_text(input_path)
    words: list[str] = pretokenize(text, special_tokens)
    vocab: dict[int, bytes] = initialize_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    
    token_sequences: dict[tuple[bytes], int] = {}
    for word in words:
        if word in special_tokens:
            continue
        
        # print(word)
        token_tuple = tuple(
            bytes([b])
            for b in word.encode("utf-8")
        )
        token_sequences[token_tuple] = token_sequences.get(token_tuple, 0) + 1
    
    while len(vocab) < vocab_size:
        pair_counts = count_pairs(token_sequences)
        if not pair_counts:
            break
        
        best_pair = find_best_pair(pair_counts)
        merges.append(best_pair)
        
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        token_sequences = apply_merge(token_sequences, best_pair)
    
    return (vocab, merges)


if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path = "data/sample.txt",
        vocab_size = 257 + 10,
        special_tokens = ["<|endoftext|>"],
    )
    print("Vocabulary:")
    for idx, token in vocab.items():
        print(f"{idx}: {token}")
    print("\nMerges:")
    for merge in merges:
        print(merge)