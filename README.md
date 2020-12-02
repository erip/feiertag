# Feiertag

![Build](https://github.com/erip/feiertag/workflows/build/badge.svg)

Feiertag is an open-source neural sequence tagging toolkit built with PyTorch.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install feiertag.

```bash
git clone git@github.com:erip/feiertag.git
cd feiertag
pip install -e .
```

## Usage

Feiertag supports a handful of baseline architectures and datasets with configuration handled by [Hydra](http://hydra.cc/). An example to train an NER model on CoNLL 2003 formatted data is shown below:

```sh
feiertag-train data.train="$DATA_DIR/train.txt" data.valid="$DATA_DIR/valid.txt" data_format=conll2003 trainer.max_epochs=25 embedding.path="$EMBEDDING_DIR/glove.6B.50d.txt" model=bilstm_crf data.loader.batch_size=128 embedding.freeze=false
```

Model training is handled by [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and kwargs passed to the `pytorch_lightning.Trainer` class can be provided by Hydra override syntax. For example, to emulate `pl.Trainer(gpus='0,1', deterministic=True)`, use the following syntax:


```sh
feiertag-train ... +trainer.gpus="0,1" +trainer.deterministic=true
```

## Managing Experiments

By default, Hydra maintains an output directory structure to separate runs. Similarly, PyTorch-Lightning logs checkpoints, hyperparameters, and [Tensorboard](https://www.tensorflow.org/tensorboard/) logfiles with the following structure:

```tree
outputs
└── 2020-12-02
    └── 18-02-11
        ├── lightning_logs
        │   └── version_0
        │       ├── checkpoints
        │       │   └── epoch=1.ckpt
        │       ├── events.out.tfevents.1606950132.erip.16601.0
        │       └── hparams.yaml
        └── train.log
```

To view the above experiment in tensorboard, issue `tensorboard --logdir outputs/2020-12-02/18-02-11/lightning_logs/version_0/`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
