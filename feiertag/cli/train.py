#!/usr/bin/env python3

import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


from feiertag.data.embeddings import read_glove_embedding
from feiertag.data.vocab import Vocab
from feiertag.data.dataloaders import PaddedDataLoader
from feiertag.types import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import os

from typing import Tuple, Any


def setup_train_data(
    vocab_config: Any, dataset_config: Any, train_file: Path, dev_file: Path, **kwargs
) -> Tuple[Vocab, Vocab, DataLoader, DataLoader]:
    vocab_reader = hydra.utils.instantiate(vocab_config)
    word_vocab, tag_vocab = vocab_reader.read_vocabs(hydra.utils.to_absolute_path(train_file))
    train_ds = hydra.utils.instantiate(
        dataset_config, hydra.utils.to_absolute_path(train_file), word_vocab, tag_vocab
    )
    dev_ds = hydra.utils.instantiate(dataset_config, hydra.utils.to_absolute_path(dev_file), word_vocab, tag_vocab)
    train_loader = PaddedDataLoader(
        train_ds, word_vocab.pad_index, tag_vocab.pad_index, **kwargs
    )
    # Don't shuffle the dev set
    kwargs.pop("shuffle", None)
    dev_loader = PaddedDataLoader(
        dev_ds, word_vocab.pad_index, tag_vocab.pad_index, **kwargs
    )
    return word_vocab, tag_vocab, train_loader, dev_loader


def setup_embedding(embedding_config: Any, word_vocab: Vocab) -> nn.Embedding:
    embedding_file = embedding_config["path"]
    embedding_dim = embedding_config["embedding_dim"]

    if embedding_file:
        print("Reading embedding...")
        embedding = read_glove_embedding(
            hydra.utils.to_absolute_path(embedding_file),
            word_vocab,
            embedding_dim=embedding_dim,
            freeze=embedding_config["freeze"],
        )
    else:
        print("Generating default embeddings...")
        embedding = nn.Embedding(
            len(word_vocab), embedding_dim, padding_idx=word_vocab.pad_index
        )
    return embedding


@hydra.main(config_path=os.path.join("..", "config"), config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg["random_seed"])

    word_vocab, tag_vocab, train_loader, dev_loader = setup_train_data(
        cfg["data_format"]["vocab"],
        cfg["data_format"]["dataset"],
        cfg["data"]["train"],
        cfg["data"]["valid"],
        **cfg["data"]["loader"],
    )

    print(f"word vocab size: {len(word_vocab)}, tagset size: {len(tag_vocab)}")

    embedding = setup_embedding(cfg["embedding"], word_vocab)

    print("Instantiating model...")
    model = hydra.utils.instantiate(cfg["model"], word_vocab, tag_vocab, embedding)
    print(model)

    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    trainer = pl.Trainer(**cfg["trainer"], logger=tb_logger)

    trainer.fit(model, train_loader, dev_loader)
