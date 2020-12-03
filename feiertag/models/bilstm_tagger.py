import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from feiertag.data.vocab import Vocab
from feiertag.models.feiertag_model import FeiertagModel

from pytorch_lightning.metrics import functional as FM


class BiLSTMTagger(FeiertagModel):
    def __init__(
        self,
        word_vocab: Vocab,
        tag_vocab: Vocab,
        embedding: nn.Embedding,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__(word_vocab, tag_vocab)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.embedding = embedding
        self.lstm = nn.LSTM(
            embedding.embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
        )

        # Maps the output of the LSTM into tag space.
        self.fc = nn.Linear(hidden_dim * 2, self.tagset_size)

        self.f1 = pl.metrics.classification.F1()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, **kwargs):
        # sentences, tags = [seq len, batch size]
        sentences, tags = batch

        # pass text through embedding layer
        # embedded = [seq len, batch size, embed dim]
        embedded = self.dropout(self.embedding(sentences))

        # pass embeddings into LSTM
        # outputs = [seq len, batch size, hidden layer * n dir]
        outputs, _ = self.lstm(embedded)

        # we use our outputs to make a prediction of what the tag should be
        # preds = [seq len, batch size, output_dim]
        preds = self.fc(self.dropout(outputs)).permute(0, 2, 1)

        pred_tags = self(sentences)

        loss = F.cross_entropy(preds, tags, ignore_index=self.tag_vocab.pad_index)

        self.log("train_f1", self.f1(pred_tags, tags), on_step=True)

        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log("train_f1", self.f1.compute(), on_epoch=True)

    def validation_step(self, batch, batch_idx, **kwargs):
        sentences, tags = batch

        pred_tags = self(sentences)
        embedded = self.embedding(sentences)
        outputs, _ = self.lstm(embedded)
        preds = self.fc(outputs).permute(0, 2, 1)

        loss = F.cross_entropy(preds, tags)

        self.log("val_f1", self.f1(pred_tags, tags), on_step=True)

        return loss

    def validation_step_end(self, outs):
        # log epoch metric
        self.log("val_f1", self.f1.compute(), on_epoch=True)

    def forward(self, sentences):
        embedded = self.embedding(sentences)
        outputs, _ = self.lstm(embedded)
        return self.fc(outputs).argmax(dim=2).t()

    def test_step(self, batch, batch_idx, **kwargs):
        sentences, tags = batch

        embedded = self.embedding(sentences)
        outputs, _ = self.lstm(embedded)

        pred_tags = self(sentences).t()

        f1 = (FM.f1(pred_tags, tags, num_classes=self.tagset_size),)

        metrics = {"f1": f1}
        self.log_dict(metrics)

    def configure_optimizers(self):
        return optim.Adam(self.parameters())
