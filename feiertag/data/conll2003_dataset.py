import torch

from typing import List, Callable, Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset

from feiertag.types import Path
from feiertag.data.vocab import Vocab, VocabReader


@dataclass
class CoNLL2003Token:
    form: str
    pos: str
    phrase_tag: str
    entity_tag: str


@dataclass
class CoNLL2003Example:
    tokens: List[CoNLL2003Token]

    @classmethod
    def from_lines(cls, lines: List[str]):
        return cls([CoNLL2003Token(*line.strip().split()) for line in lines])


class CoNLL2003Dataset(Dataset):
    def __init__(
        self,
        examples: List[CoNLL2003Example],
        word_vocab: Vocab,
        tag_vocab: Vocab,
        tag_extractor: Callable[[CoNLL2003Token], str],
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
        ner = torch.LongTensor(
            [self._tag_vocab.bos_index]
            + [self._tag_vocab[self._extractor(tok)] for tok in ex.tokens]
            + [self._tag_vocab.eos_index]
        )
        return tokens, ner

    @classmethod
    def from_file(
        cls,
        file: Path,
        word_vocab: Vocab,
        tag_vocab: Vocab,
        tag_extractor: Callable[[CoNLL2003Example], str],
        **kwargs,
    ) -> "CoNLL2003Dataset":
        with open(file, **kwargs) as f:
            return cls(
                [
                    CoNLL2003Example.from_lines(
                        [line.strip() for line in s.splitlines()]) for s in f.read().strip().split("\n\n") if
                    s.strip() != "-DOCSTART- -X- -X- O"
                ],
                word_vocab,
                tag_vocab,
                tag_extractor,
            )


class CoNLL2003NERDataset(CoNLL2003Dataset):
    def __init__(
        self,
        examples: List[CoNLL2003Example],
        word_vocab: Vocab,
        tag_vocab: Vocab,
        *args,
    ):
        super().__init__(examples, word_vocab, tag_vocab, lambda tok: tok.entity_tag)

    @classmethod
    def from_file(
        cls, file: Path, word_vocab: Vocab, tag_vocab: Vocab, **kwargs
    ) -> CoNLL2003Dataset:
        return super().from_file(
            file, word_vocab, tag_vocab, lambda t: t.entity_tag, **kwargs
        )


class CoNLL2003VocabReader(VocabReader):
    @staticmethod
    def __read_conll2003_vocabs(file: Path, *args, **kwargs) -> Tuple[Vocab, Vocab]:
        word_vocab, tag_vocab = Vocab(), Vocab()
        with open(file, **kwargs) as f:
            for line in f:
                # Skip docstarts and empty lines
                if not line.strip() or line.strip() == "-DOCSTART- -X- -X- O":
                    continue
                tok = CoNLL2003Token(*line.strip().split())
                word_vocab += tok.form
                tag_vocab += tok.entity_tag
        return word_vocab, tag_vocab

    def read_vocabs(self, file: Path, *args, **kwargs) -> Tuple[Vocab, Vocab]:
        return self.__read_conll2003_vocabs(file, *args, **kwargs)
