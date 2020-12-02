import pytest

from feiertag.data.conll2003_dataset import (
    CoNLL2003Token,
    CoNLL2003Example,
    CoNLL2003Dataset,
    CoNLL2003NERDataset,
    CoNLL2003VocabReader,
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


def test_unknown_token_in_example_yields_unk():
    word_vocab = Vocab()
    tag_vocab = Vocab()
    form = "Pierre"
    entity_tag = "B-PER"
    assert form not in word_vocab
    tag_vocab += entity_tag
    examples = [CoNLL2003Example([CoNLL2003Token(form, "", "", entity_tag)])]
    ds = CoNLL2003Dataset(examples, word_vocab, tag_vocab, lambda t: t.entity_tag)
    (token_vector, tag_vector) = ds[0]
    assert word_vocab.unk_index in token_vector
    assert tag_vocab.unk_index not in tag_vector


def test_unknown_token_and_tag_in_example_yields_unk():
    word_vocab = Vocab()
    tag_vocab = Vocab()
    form = "Pierre"
    entity_tag = "B-PER"
    assert form not in word_vocab
    assert entity_tag not in tag_vocab
    examples = [CoNLL2003Example([CoNLL2003Token(form, "", "", entity_tag)])]
    ds = CoNLL2003Dataset(examples, word_vocab, tag_vocab, lambda t: t.entity_tag)
    (token_vector, tag_vector) = ds[0]
    assert word_vocab.unk_index in token_vector
    assert tag_vocab.unk_index in tag_vector


def test_unknown_token_in_example_yields_unk_udposdataset():
    word_vocab = Vocab()
    tag_vocab = Vocab()
    form = "Pierre"
    entity_tag = "B-PER"
    assert form not in word_vocab
    tag_vocab += entity_tag
    examples = [CoNLL2003Example([CoNLL2003Token(form, "", "", entity_tag)])]
    ds = CoNLL2003NERDataset(examples, word_vocab, tag_vocab)
    (token_vector, tag_vector) = ds[0]
    assert word_vocab.unk_index in token_vector
    assert tag_vocab.unk_index not in tag_vector


def test_unknown_token_and_tag_in_example_yields_unk_udposdataset():
    word_vocab = Vocab()
    tag_vocab = Vocab()
    form = "Pierre"
    entity_tag = "B-PER"
    assert form not in word_vocab
    assert entity_tag not in tag_vocab
    examples = [CoNLL2003Example([CoNLL2003Token(form, "", "", entity_tag)])]
    ds = CoNLL2003NERDataset(examples, word_vocab, tag_vocab)
    (token_vector, tag_vector) = ds[0]
    assert word_vocab.unk_index in token_vector
    assert tag_vocab.unk_index in tag_vector


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


def test_conll2003_vocab_reader(tmp_path, example_dataset):
    file = tmp_path / "tmp.txt"

    with open(file, "w", encoding="utf-8") as f:
        f.write(example_dataset)

    word_vocab, tag_vocab = CoNLL2003VocabReader().read_vocabs(file, encoding="utf-8")

    words = []
    entity_tags = []
    for line in example_dataset.splitlines():
        if line.strip() and line.strip() != "-DOCSTART- -X- -X- O":
            parts = line.split()
            words.append(parts[0])
            entity_tags.append(parts[3])

    assert all(word in word_vocab for word in words)
    assert all(tag in tag_vocab for tag in entity_tags)
