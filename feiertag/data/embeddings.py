import torch
import torch.nn as nn

from feiertag.types import Path
from feiertag.data.vocab import Vocab


def read_glove_embedding(
    file: Path, word_vocab: Vocab, freeze: bool, embedding_dim: int, **kwargs
) -> nn.Embedding:
    e = torch.zeros((len(word_vocab), embedding_dim), dtype=torch.float32)

    with open(file, **kwargs) as f:
        for line in f:
            word, *weights = line.strip().split()
            weights = torch.FloatTensor(list(map(float, weights)))
            idx = word_vocab[word.lower()]
            e[idx] += weights
    return nn.Embedding.from_pretrained(
        e, padding_idx=word_vocab.pad_index, freeze=freeze
    )
