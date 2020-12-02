import pytest

from feiertag.data.conll2003_dataset import (
    CoNLL2003Token,
    CoNLL2003Example,
    CoNLL2003Dataset,
    CoNLL2003NERDataset,
)
from feiertag.data.vocab import Vocab

from hypothesis import given, strategies as st


@pytest.fixture
def conll2003_example():
    return """EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O"""


@pytest.fixture
def example_dataset():
    return """-DOCSTART- -X- -X- O

EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O

Peter NNP B-NP B-PER
Blackburn NNP I-NP I-PER
"""


def test_conll2003_example_from_lines(conll2003_example):
    # Skip comments
    example = CoNLL2003Example.from_lines(conll2003_example.splitlines())
    assert len(conll2003_example.splitlines()) == len(example.tokens)


def test_conll2003_token(conll2003_example):
    tokens = conll2003_example.splitlines()
    first_token = tokens[0].split()
    tok = CoNLL2003Token(*first_token)
    for i, f in enumerate((lambda t: t.form, lambda t: t.pos, lambda t: t.phrase_tag, lambda t: t.entity_tag)):
        assert f(tok) == first_token[i]


@given(st.lists(st.from_type(CoNLL2003Example), max_size=50))
def test_uddataset_len_is_len_examples(examples):
    word_vocab = Vocab()
    tag_vocab = Vocab()
    ds = CoNLL2003NERDataset(examples, word_vocab, tag_vocab, lambda t: t.entity_tag)
    assert len(ds) == len(ds)


@pytest.mark.parametrize("cls,func", [(CoNLL2003Dataset, lambda _: ""), (CoNLL2003NERDataset, None)])
def test_conll2003dataset_from_file(tmp_path, example_dataset, cls, func):
    file = tmp_path / "tmp.txt"
    word_vocab, tag_vocab = Vocab(), Vocab()

    with open(file, "w", encoding="utf-8") as f:
        f.write(example_dataset)

    if func:
        ds = cls.from_file(file, word_vocab, tag_vocab, func, encoding="utf-8")
    else:
        ds = cls.from_file(file, word_vocab, tag_vocab, encoding="utf-8")

    assert len(ds) == 2
