import pytest

from feiertag.data.ud_dataset import (
    UDDataset,
    UDUPOSDataset,
    CoNLLUExample,
    CoNLLUToken,
)
from feiertag.data.vocab import Vocab

from hypothesis import given, strategies as st


@pytest.fixture
def example_lines():
    return [
        "# sent_id = dev-s1",
        "# text = 제일 가까운 스타벅스가 어디 있지",
        "1	제일	제일	ADV	NNG	_	2	advmod	_	_",
        "2	가까운	가깝+ㄴ	ADJ	VA+ETM	_	3	amod	_	_",
        "3	스타벅스가	스타벅스+가	NOUN	NNG+JKS	_	5	nsubj	_	_",
        "4	어디	어디	ADV	NP	_	5	advmod	_	_",
        "5	있지	있+지	ADJ	VV+EC	_	0	root	_	_",
    ]


@pytest.fixture
def example_dataset():
    return """# sent_id = dev-s1
# text = 제일 가까운 스타벅스가 어디 있지
1	제일	제일	ADV	NNG	_	2	advmod	_	_
2	가까운	가깝+ㄴ	ADJ	VA+ETM	_	3	amod	_	_
3	스타벅스가	스타벅스+가	NOUN	NNG+JKS	_	5	nsubj	_	_
4	어디	어디	ADV	NP	_	5	advmod	_	_
5	있지	있+지	ADJ	VV+EC	_	0	root	_	_

# sent_id = dev-s2
# text = 이 행사는 도가 주관하고 경기관광공사와 경기국제의료관광협의회(e-gima.com)가 주최하며 도내 8개 병원이 함께 참여한다.
1	이	이	DET	MM	_	2	det	_	_
2	행사는	행사+는	NOUN	NNG+JX	_	4	obj	_	_
3	도가	도+가	NOUN	NNG+JKS	_	4	nsubj	_	_
4	주관하고	주관+하+고	VERB	NNG+XSV+EC	_	0	root	_	_
5	경기관광공사와	경기+관광공사+와	NOUN	NNG+NNG+JC	_	11	nsubj	_	_
6	경기국제의료관광협의회	경기+국제+의료+관광+협의회	NOUN	NNG+NNG+NNG+NNG+NNG	_	5	conj	_	SpaceAfter=No
7	(	(	PUNCT	SS	_	8	punct	_	SpaceAfter=No
8	e-gima.com	e-gima.com	NOUN	SL	_	6	appos	_	SpaceAfter=No
9	)	)	PUNCT	SS	_	8	punct	_	SpaceAfter=No
10	가	가	ADP	JKS	_	6	nsubj	_	_
11	주최하며	주최+하+며	VERB	NNG+XSV+EC	_	4	conj	_	_
12	도내	도+내	NOUN	NNG+NNB	_	16	nsubj	_	_
13	8개	8+개	NOUN	SN+NNB	_	12	flat	_	_
14	병원이	병원+이	NOUN	NNG+JKS	_	12	flat	_	_
15	함께	함께	ADV	MAG	_	16	advmod	_	_
16	참여한다	참여+하+ㄴ다	VERB	NNG+XSV+EF	_	4	conj	_	SpaceAfter=No
17	.	.	PUNCT	SF	_	16	punct	_	_

"""


def test_conllu_example_from_lines(example_lines):
    # Skip comments
    lines_without_comments = [
        line for line in example_lines if not line.strip().startswith("#")
    ]
    example = CoNLLUExample.from_lines(example_lines)
    assert len(example.tokens) == len(lines_without_comments)
    # Can recreate tokens from uncommented lines
    assert [
        CoNLLUToken(*line.strip().split("\t")) for line in lines_without_comments
    ] == example.tokens


def test_unknown_token_in_example_yields_unk():
    word_vocab = Vocab()
    tag_vocab = Vocab()
    form = "Pierre"
    pos = "NNP"
    assert form not in word_vocab
    tag_vocab += pos
    examples = [CoNLLUExample([CoNLLUToken(1, form, "", pos, "", "", 0, "", "", "")])]
    ds = UDDataset(examples, word_vocab, tag_vocab, lambda t: t.upos)
    (token_vector, tag_vector) = ds[0]
    assert word_vocab.unk_index in token_vector
    assert tag_vocab.unk_index not in tag_vector


def test_unknown_token_and_tag_in_example_yields_unk():
    word_vocab = Vocab()
    tag_vocab = Vocab()
    form = "Pierre"
    pos = "NNP"
    assert form not in word_vocab
    assert pos not in tag_vocab
    examples = [CoNLLUExample([CoNLLUToken(1, form, "", pos, "", "", 0, "", "", "")])]
    ds = UDDataset(examples, word_vocab, tag_vocab, lambda t: t.upos)
    (token_vector, tag_vector) = ds[0]
    assert word_vocab.unk_index in token_vector
    assert tag_vocab.unk_index in tag_vector


def test_unknown_token_in_example_yields_unk_udposdataset():
    word_vocab = Vocab()
    tag_vocab = Vocab()
    form = "Pierre"
    pos = "NNP"
    assert form not in word_vocab
    tag_vocab += pos
    examples = [CoNLLUExample([CoNLLUToken(1, form, "", pos, "", "", 0, "", "", "")])]
    ds = UDUPOSDataset(examples, word_vocab, tag_vocab)
    (token_vector, tag_vector) = ds[0]
    assert word_vocab.unk_index in token_vector
    assert tag_vocab.unk_index not in tag_vector


def test_unknown_token_and_tag_in_example_yields_unk_udposdataset():
    word_vocab = Vocab()
    tag_vocab = Vocab()
    form = "Pierre"
    pos = "NNP"
    assert form not in word_vocab
    assert pos not in tag_vocab
    examples = [CoNLLUExample([CoNLLUToken(1, form, "", pos, "", "", 0, "", "", "")])]
    ds = UDUPOSDataset(examples, word_vocab, tag_vocab)
    (token_vector, tag_vector) = ds[0]
    assert word_vocab.unk_index in token_vector
    assert tag_vocab.unk_index in tag_vector


@given(st.lists(st.from_type(CoNLLUExample), max_size=50))
def test_uddataset_len_is_len_examples(examples):
    word_vocab = Vocab()
    tag_vocab = Vocab()
    ds = UDDataset(examples, word_vocab, tag_vocab, lambda t: t.upos)
    assert len(ds) == len(ds)


@pytest.mark.parametrize("cls,func", [(UDDataset, lambda _: ""), (UDUPOSDataset, None)])
def test_uddataset_from_file(tmp_path, example_dataset, cls, func):
    file = tmp_path / "tmp.txt"
    word_vocab, tag_vocab = Vocab(), Vocab()

    with open(file, "w", encoding="utf-8") as f:
        f.write(example_dataset)

    if func:
        ds = cls.from_file(file, word_vocab, tag_vocab, func)
    else:
        ds = cls.from_file(file, word_vocab, tag_vocab)

    assert len(ds) == 2
