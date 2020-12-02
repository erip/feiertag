import torch

from feiertag.models.feiertag_model import FeiertagModel
from feiertag.data.vocab import Vocab
from feiertag.types import Tokens

import pytest


class DummyModel(FeiertagModel):
    def __init__(self, word_vocab: Vocab, tag_vocab: Vocab):
        super().__init__(word_vocab, tag_vocab)

    def _cheat_tag(self, sent: Tokens):
        evens = {"0", "2", "4", "6", "8"}
        # Just look up the right answer.
        return [
            self.tag_vocab["EVEN"]
            if self.word_vocab.token(tok.item()) in evens
            else self.tag_vocab["ODD"]
            for tok in sent
        ]

    def forward(self, sentences):
        return torch.LongTensor([self._cheat_tag(sent) for sent in sentences.t()])


@pytest.fixture
def word_vocab():
    word_vocab_ = Vocab()
    for i in range(11):
        word_vocab_ += str(i)
    return word_vocab_


@pytest.fixture
def tag_vocab():
    tag_vocab_ = Vocab()
    tag_vocab_ += "EVEN"
    tag_vocab_ += "ODD"
    return tag_vocab_


def test_tagging(word_vocab, tag_vocab):
    model = DummyModel(word_vocab, tag_vocab)
    input = ["0", "0", "1"]
    tagged = model.tag(input)
    expected_output = [[("0", "EVEN"), ("0", "EVEN"), ("1", "ODD")]]
    assert tagged == expected_output
