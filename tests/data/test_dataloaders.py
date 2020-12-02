import pytest

import torch

from torch.utils.data import Dataset
from feiertag.data.dataloaders import PaddedDataLoader


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


@pytest.fixture
def tag_pad_index():
    return -1


@pytest.fixture
def word_pad_index():
    return 0


def test_sample_dataloader(tag_pad_index, word_pad_index):

    data = [
        (torch.LongTensor([1, 1, 1]), torch.LongTensor([2, 2, 2])),
        (torch.LongTensor([1, 1]), torch.LongTensor([2, 2]))
    ]

    num_examples = len(data)
    max_size = max(t[0].shape[0] for t in data)

    print(max_size)

    ds = DummyDataset(data)

    loader = PaddedDataLoader(ds, word_pad_index, tag_pad_index, batch_size=2)
    (word_idx, tag_idx) = next(iter(loader))

    # Shorter example gets padded with the appropriate pad index
    assert word_idx[max_size-1, num_examples-1] == word_pad_index
    assert tag_idx[max_size-1, num_examples-1] == tag_pad_index
