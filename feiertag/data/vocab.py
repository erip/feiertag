from collections import Counter

from abc import ABCMeta, abstractmethod
from typing import Tuple

from feiertag.types import Path


class Vocab:
    def __init__(
        self,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        self._t2i = {}
        self._i2t = {}
        self._counts = Counter()
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        for tok, _id in {
            unk_token: 0,
            pad_token: 1,
            bos_token: 2,
            eos_token: 3,
        }.items():
            self._t2i[tok] = _id
            self._i2t[_id] = tok
            self._counts[tok] += 1

        self.unk_index = self._t2i[self.unk_token]
        self.pad_index = self._t2i[self.pad_token]
        self.bos_index = self._t2i[self.bos_token]
        self.eos_index = self._t2i[self.eos_token]

    def __len__(self):
        return len(self._t2i)

    def __contains__(self, item: str) -> bool:
        return item in self._t2i

    def __iadd__(self, tok: str) -> "Vocab":
        if tok not in self._counts:
            current_max_id = len(self._t2i)
            self._t2i[tok] = current_max_id
            self._i2t[current_max_id] = tok
            self._counts[tok] += 1
        return self

    def __getitem__(self, tok: str) -> int:
        return self._t2i.get(tok, self.unk_index)

    def token(self, idx: int) -> str:
        return self._i2t.get(idx, self.unk_token)

    def add_file(self, file: Path, **kwargs) -> None:
        with open(file, **kwargs) as f:
            for line in f:
                for tok in line.strip().split():
                    self.__iadd__(tok)


class VocabReader(metaclass=ABCMeta):
    @abstractmethod
    def read_vocabs(self, file: Path, *args, **kwargs) -> Tuple[Vocab, Vocab]:
        pass
