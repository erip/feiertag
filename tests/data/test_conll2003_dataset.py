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