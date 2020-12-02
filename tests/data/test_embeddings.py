import pytest

from feiertag.data.vocab import Vocab
from feiertag.data.embeddings import read_glove_embedding


@pytest.fixture
def example_embedding():
    return ["test 0.1 0.2 -0.3"]


@pytest.fixture
def empty_vocab(example_embedding):
    return Vocab()


@pytest.fixture
def nonempty_vocab(example_embedding):
    vocab = Vocab()
    vocab += example_embedding[0].split()[0]
    return vocab


def test_read_glove_embedding_nonempty_vocab(
    tmp_path, example_embedding, nonempty_vocab
):
    file = tmp_path / "embedding.txt"

    with open(file, "w", encoding="utf-8") as f:
        for line in example_embedding:
            print(line, file=f)

    read_embedding = read_glove_embedding(
        file, word_vocab=nonempty_vocab, embedding_dim=3, freeze=True
    )
    assert len(nonempty_vocab) == read_embedding.num_embeddings
    assert read_embedding.embedding_dim == 3


def test_read_glove_embedding_empty_vocab(tmp_path, example_embedding, empty_vocab):
    file = tmp_path / "embedding.txt"

    with open(file, "w", encoding="utf-8") as f:
        for line in example_embedding:
            print(line, file=f)

    read_embedding = read_glove_embedding(
        file, word_vocab=empty_vocab, embedding_dim=3, freeze=True
    )
    assert len(empty_vocab) == read_embedding.num_embeddings
    assert read_embedding.embedding_dim == 3
