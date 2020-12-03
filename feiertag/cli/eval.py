import os
import warnings

# PyTorch-Lightning warns amount mismatches in num classes at the batch-level, so we suppress these.
warnings.filterwarnings(
    "ignore", module="pytorch_lightning.utilities.distributed", category=RuntimeWarning
)

import hydra

from torch.utils.data import DataLoader

from typing import Any
from omegaconf import DictConfig, OmegaConf

from feiertag.types import Path
from feiertag.data.vocab import Vocab
from feiertag.data.dataloaders import PaddedDataLoader

import pytorch_lightning as pl


def setup_eval_data(
    word_vocab: Vocab, tag_vocab: Vocab, dataset_config: Any, test_file: Path, **kwargs
) -> DataLoader:

    test_ds = hydra.utils.instantiate(
        dataset_config, hydra.utils.to_absolute_path(test_file), word_vocab, tag_vocab
    )

    # Don't shuffle the dev set
    kwargs.pop("shuffle", None)

    test_loader = PaddedDataLoader(
        test_ds, word_vocab.pad_index, tag_vocab.pad_index, **kwargs
    )

    return test_loader


@hydra.main(config_path=os.path.join("..", "config"), config_name="eval")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model = hydra.utils.instantiate(
        cfg.model.load, hydra.utils.to_absolute_path(cfg.path)
    )
    print(model)

    test_loader = setup_eval_data(
        model.word_vocab,
        model.tag_vocab,
        cfg["data_format"]["dataset"],
        cfg["data"]["test"],
    )

    trainer = pl.Trainer() if "trainer" not in cfg else pl.Trainer(**cfg["trainer"])
    trainer.test(model=model, test_dataloaders=test_loader)
