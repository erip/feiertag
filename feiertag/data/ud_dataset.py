from typing import List, Tuple, Callable

import torch
from dataclasses import dataclass
from torch.utils.data import Dataset

from feiertag.data.vocab import Vocab, VocabReader
from feiertag.types import Path


@dataclass
class CoNLLUToken:
    _id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str


@dataclass
class CoNLLUExample:
    tokens: List[CoNLLUToken]

    @classmethod
    def from_lines(cls, lines: List[str]):
        return cls(
            [
                CoNLLUToken(*line.strip().split("\t"))
                for line in lines
                if not line.strip().startswith("#")
            ]
        )


class UDDataset(Dataset):
    def __init__(
        self,
        examples: List[CoNLLUExample],
        word_vocab: Vocab,
        tag_vocab: Vocab,
        tag_extractor: Callable[[CoNLLUToken], str],
    ):
        self._word_vocab = word_vocab
        self._tag_vocab = tag_vocab
        self._examples = examples
        self._extractor = tag_extractor

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        ex = self._examples[idx]
        tokens = torch.LongTensor(
            [self._word_vocab.bos_index]
            + [self._word_vocab[tok.form] for tok in ex.tokens]
            + [self._word_vocab.eos_index]
        )
        upos = torch.LongTensor(
            [self._tag_vocab.bos_index]
            + [self._tag_vocab[self._extractor(tok)] for tok in ex.tokens]
            + [self._tag_vocab.eos_index]
        )
        return tokens, upos

    @classmethod
    def from_file(
        cls,
        file: Path,
        word_vocab: Vocab,
        tag_vocab: Vocab,
        tag_extractor: Callable[[CoNLLUToken], str],
        **kwargs
    ) -> "UDDataset":
        with open(file, **kwargs) as f:
            return cls(
                [
                    CoNLLUExample.from_lines(list(map(str.strip, s.splitlines())))
                    for s in f.read().strip().split("\n\n")
                ],
                word_vocab,
                tag_vocab,
                tag_extractor,
            )


class UDUPOSDataset(UDDataset):
    def __init__(
        self, examples: List[CoNLLUExample], word_vocab: Vocab, tag_vocab: Vocab, *args
    ):
        super().__init__(examples, word_vocab, tag_vocab, lambda tok: tok.upos)

    @classmethod
    def from_file(
        cls, file: Path, word_vocab: Vocab, tag_vocab: Vocab, **kwargs
    ) -> UDDataset:
        return super().from_file(
            file, word_vocab, tag_vocab, lambda t: t.upos, **kwargs
        )


class CoNLLUVocabReader(VocabReader):
    @staticmethod
    def __read_conllu_vocabs(file: Path, *args, **kwargs) -> Tuple[Vocab, Vocab]:
        word_vocab, tag_vocab = Vocab(), Vocab()
        with open(file, **kwargs) as f:
            for line in f:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                tok = CoNLLUToken(*line.strip().split("\t"))
                word_vocab += tok.form
                tag_vocab += tok.upos
        return word_vocab, tag_vocab

    def read_vocabs(self, file: Path, *args, **kwargs) -> Tuple[Vocab, Vocab]:
        return self.__read_conllu_vocabs(file, *args, **kwargs)
