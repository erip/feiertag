import torch

import pytorch_lightning as pl

from feiertag.data.vocab import Vocab

from feiertag.types import Tokens, TaggedSentences
from torch.nn.utils.rnn import pad_sequence


class FeiertagModel(pl.LightningModule):
    def __init__(self, word_vocab: Vocab, tag_vocab: Vocab):
        super().__init__()
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.vocab_size = len(self.word_vocab)
        self.tagset_size = len(self.tag_vocab)

    def tag(self, tokens: Tokens) -> TaggedSentences:
        assert len(tokens) > 0
        # Handles tagging a list of str
        if isinstance(tokens[0], str):
            return self.tag([tokens])

        token_ids = [
            (
                torch.LongTensor(
                    [self.word_vocab.bos_index]
                    + [self.word_vocab[tok] for tok in toks]
                    + [self.tag_vocab.eos_index]
                )
            )
            for toks in tokens
        ]

        token_ids = pad_sequence(token_ids, padding_value=self.word_vocab.pad_index)

        preds = self(token_ids)

        tag_names = [
            [
                self.tag_vocab.token(idx.item())
                for i, idx in enumerate(pred)
                # Chop off BOS and EOS bookends
            ][1:-1]
            for pred in preds
        ]

        assert len(tag_names) == len(tokens) and all(
            len(tag) == len(tok) for tag, tok in zip(tag_names, tokens)
        )

        return [
            [(word, tag) for word, tag in zip(words, tags)]
            for words, tags in zip(tokens, tag_names)
        ]
