import torch

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from typing import Tuple

from feiertag.types import Batch


def pad_collate(
    batch: Batch, word_pad_idx: int, tag_pad_idx: int
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    data = pad_sequence([item[0] for item in batch], padding_value=word_pad_idx)
    target = pad_sequence([item[1] for item in batch], padding_value=tag_pad_idx)
    return data, target


def PaddedDataLoader(
    dataset: Dataset, word_pad_index: int, tag_pad_index: int, **kwargs
) -> DataLoader:
    return DataLoader(
        dataset,
        collate_fn=lambda b: pad_collate(b, word_pad_index, tag_pad_index),
        **kwargs
    )
