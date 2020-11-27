from io import StringIO

from feiertag.data.vocab import Vocab


def test_default_special_tokens():
    unk_token = "<unk>"
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    vocab = Vocab()
    assert vocab[unk_token] == 0
    assert vocab[pad_token] == 1
    assert vocab[bos_token] == 2
    assert vocab[eos_token] == 3


def test_override_special_tokens():
    unk_token = "[UNK]"
    pad_token = "[PAD]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    vocab = Vocab(unk_token, pad_token, bos_token, eos_token)
    assert vocab[unk_token] == 0
    assert vocab[pad_token] == 1
    assert vocab[bos_token] == 2
    assert vocab[eos_token] == 3


def test_add_token():
    vocab = Vocab()
    tok = "hello"
    vocab += tok

    assert vocab[tok] == 4


def test_unknown_retrieval_yields_unk():
    vocab = Vocab()

    assert vocab["hello"] == vocab["<unk>"]


def test_from_file(tmp_path):
    t1, t2, t3, t4 = "hello", ",", "world", "!"
    s = f"{t1} {t2} {t3} {t4}"
    filename = tmp_path / "tmp.txt"
    with open(filename, "w") as f:
        print(s, file=f)

    vocab = Vocab()
    vocab.add_file(filename)

    for i, e in enumerate([t1, t2, t3, t4]):
        # Offset by four because of unk, pad, bos, eos tokens
        assert vocab[e] == i + 4

    assert len(vocab) == 8
