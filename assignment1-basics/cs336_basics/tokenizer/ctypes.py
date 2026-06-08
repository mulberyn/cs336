from typing import TypeAlias

Vocabulary: TypeAlias = dict[int, bytes]

MergeRule: TypeAlias = tuple[bytes, bytes]

CorpusTokens: TypeAlias = list[list[bytes]]