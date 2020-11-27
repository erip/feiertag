import os
import pathlib
from typing import Union, TypeVar, List, Tuple

Path = Union[str, os.PathLike, pathlib.Path]

T = TypeVar("T")
Batch = List[T]
Tokens = Union[List[str], List[List[str]]]
TaggedToken = Tuple[str, str]
TaggedSentence = List[TaggedToken]
TaggedSentences = List[TaggedSentence]