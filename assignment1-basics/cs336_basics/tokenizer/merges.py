from collections import Counter


def count_pairs(
    token_sequences: dict[tuple[bytes], int]
) -> Counter[tuple[bytes, bytes]]:
    pair_counts = Counter()

    for seq, freq in token_sequences.items():
        for i in range(len(seq) - 1):
            pair_counts[(seq[i], seq[i + 1])] += freq

    return pair_counts


def find_best_pair(
    pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    if not pair_counts:
        raise ValueError("No pairs to choose from")
    
    best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
    return best_pair
    
    
def apply_merge(
    token_sequences: dict[tuple[bytes], int], 
    pair: tuple[bytes, bytes]
) -> dict[tuple[bytes], int]:
    """
    input: 
        token_sequences: dict[tuple[bytes], int], the current token sequences and their frequencies
        pair: tuple[bytes, bytes], the pair to merge
    output:
        new_token_sequences: dict[tuple[bytes], int], the updated token sequences after merging
    """
    new_token_sequences = {}
    merged_token = pair[0] + pair[1]

    for seq, freq in token_sequences.items():
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                new_seq.append(merged_token)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_token_sequences[tuple(new_seq)] = freq

    return new_token_sequences


if __name__ == "__main__":
    token_sequences = {
        (b'a', b'b', b'c'): 5,
        (b'a', b'b'): 3,
        (b'b', b'c'): 2,
    }
    pair_counts = count_pairs(token_sequences)
    print("Pair counts:", pair_counts)

    best_pair = find_best_pair(pair_counts)
    print("Best pair:", best_pair)

    new_sequences = apply_merge(token_sequences, best_pair)
    print("New sequences after merge:", new_sequences)