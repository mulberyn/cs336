from cs336_basics.tokenizer.corpus import load_text
from cs336_basics.tokenizer.pretokenize import pretokenize
from cs336_basics.tokenizer.vocabulary import (
  initialize_vocab,
)
from cs336_basics.tokenizer.merges import (
    find_best_pair,
)
from cs336_basics.tokenizer.incremental import (
    build_from_token_sequences,
    update_word,
)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text = load_text(input_path)
    words = pretokenize(text, special_tokens)
    
    vocab = initialize_vocab(special_tokens)
    token_sequences = {}
    for word in words:
        if word in special_tokens:
            continue
        
        token_tuple = tuple(word.encode("utf-8")) # (b'a', b'b', b'c') 形式
        token_sequences[token_tuple] = token_sequences.get(token_tuple, 0) + 1
    
    merges = []
    words, pair_counts, pair_to_words = build_from_token_sequences(token_sequences)
    
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        
        best_pair = find_best_pair(pair_counts, vocab)
        new_id = len(vocab)
        
        a, b = best_pair
        merges.append((vocab[a], vocab[b]))
        vocab[new_id] = vocab[a] + vocab[b]
        
        affected = list(pair_to_words.get(best_pair, []))
        for word_id in affected:
            word = words[word_id]
            if word is None:
                continue
            update_word(word, a, b, new_id, pair_counts, pair_to_words, word_id)
        
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]
    
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