import pytest

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from feiertag.data.vocab import Vocab
from feiertag.models.bilstm_tagger import BiLSTMTagger
from feiertag.models.bilstm_crf_tagger import BiLSTM_CRF_Tagger

import pytorch_lightning as pl


@pytest.fixture
def word_vocab():
    vocab = Vocab()
    for i in range(11):
        vocab += str(i)
    return vocab

@pytest.fixture
def tag_vocab():
    vocab = Vocab()
    vocab += "ODD"
    vocab += "EVEN"
    return vocab


@pytest.fixture
def train_loader(word_vocab, tag_vocab):
    data = []
    for _ in range(100):
        span = [random.randint(0, 11) for _ in range(50)]
        labels = ["EVEN" if e % 2 == 0 else "ODD" for e in span]
        data.append((span, labels))

    tensor = []
    for (span, labels) in data:
        s = torch.LongTensor([word_vocab[e] for e in span])
        l = torch.LongTensor([tag_vocab[e] for e in labels])
        tensor.append((s, l))

    return DataLoader(tensor, num_workers=4)


@pytest.fixture
def valid_loader(word_vocab, tag_vocab):
    data = []
    for _ in range(100):
        span = [random.randint(0, 11) for _ in range(50)]
        labels = ["EVEN" if e % 2 == 0 else "ODD" for e in span]
        data.append((span, labels))

    tensor = []
    for (span, labels) in data:
        s = torch.LongTensor([word_vocab[e] for e in span])
        l = torch.LongTensor([tag_vocab[e] for e in labels])
        tensor.append((s, l))

    return DataLoader(tensor, num_workers=4)


@pytest.mark.parametrize("model_cls,kwargs", [
    (BiLSTMTagger, {"embedding": nn.Embedding(10, 5), "hidden_dim": 16, "num_layers": 2}),
    (BiLSTM_CRF_Tagger, {"embedding": nn.Embedding(10, 5), "hidden_dim": 16, "num_layers": 2})
])
def test_model(model_cls, word_vocab, tag_vocab,train_loader, valid_loader, kwargs):
    model = model_cls(word_vocab=word_vocab, tag_vocab=tag_vocab, **kwargs)
    trainer = pl.Trainer(logger=False, checkpoint_callback=False, max_epochs=1)
    trainer.fit(model, train_loader, valid_loader)

