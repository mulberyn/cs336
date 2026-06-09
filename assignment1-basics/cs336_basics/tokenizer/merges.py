from collections import Counter


def count_pairs(
    token_sequences: dict[tuple[int], int]
) -> Counter[tuple[int, int]]:
    pair_counts = Counter()

    for seq, freq in token_sequences.items():
        for i in range(len(seq) - 1):
            pair_counts[(seq[i], seq[i + 1])] += freq

    return pair_counts


def find_best_pair(
    pair_counts: Counter[tuple[int, int]],
    vocab: dict[int, bytes],
) -> tuple[int, int]:
    if not pair_counts:
        raise ValueError("No pairs to choose from")
    
    best_pair = max(
        pair_counts.items(), 
        key = lambda p: (
            p[1],
            (vocab[p[0][0]], vocab[p[0][1]])
        )
    )[0]
    return best_pair
    
    
def apply_merge(
    token_sequences: dict[tuple[int], int], 
    pair: tuple[int, int],
    new_id: int,
) -> dict[tuple[int], int]:
    """
    input: 
        token_sequences: dict[tuple[int], int], the current token sequences and their frequencies
        pair: tuple[int, int], the pair to merge
    output:
        new_token_sequences: dict[tuple[int], int], the updated token sequences after merging
    """
    new_token_sequences = {}

    for seq, freq in token_sequences.items():
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_token_sequences[tuple(new_seq)] = freq

    return new_token_sequences


if __name__ == "__main__":
    token_sequences = {
        (1, 2, 3): 5,
        (1, 2): 3,
        (2, 3): 2,
    }
    pair_counts = count_pairs(token_sequences)
    print("Pair counts:", pair_counts)

    best_pair = find_best_pair(pair_counts)
    print("Best pair:", best_pair)

    new_sequences = apply_merge(token_sequences, best_pair)
    print("New sequences after merge:", new_sequences)