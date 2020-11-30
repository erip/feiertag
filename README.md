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

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
