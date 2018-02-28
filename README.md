# Conversational Banking

Implementation of models in paper **Fast and Effective Neural Conversational Banking**, if you use this code, please cite this work.

## Setup

#### 1. Install necessary libraries

With CUDA 8.0:
```bash
$ conda create --name conv-bank python=3.6.1
$ source activate conv-bank
$ conda install pytorch torchvision cuda80 -c soumith
<!-- $ pip install ipdb -->
```

Without CUDA:
```bash
$ conda create --name conv-bank python=3.6.1
$ source activate conv-bank
$ conda install pytorch torchvision -c soumith
<!-- $ pip install ipdb -->
```

#### 2. Create dataset file

```bash
$ cd data
$ tar -zxvf samantha.tar.gz
$ cd ..
$ source data_handler/create_dataset.sh
```


## Model options

* fasp
* redfasp
* bytenet
* rnn
* attn-rnn
* redmulnet-1
* redmulnet-2
* fasp --multi-linear
* redfasp --multi-linear
* bytenet --multi-linear

## Train model

Training command:
```bash
$ python train.py --model <model> <--multi-linear> --epochs <max-epochs> --save-dir <save-path>
```

To train a single model:
```bash
$ python train.py --model fasp --epochs 20 --save-dir checkpoint/samantha/fasp
```

To train all models with default parameters:
```bash
$ source train_models.sh <max-epochs>
```

## Eval model

Evaluating command:
```bash
$ source eval_models.sh <path-to-models> <path-to-data>
```

To eval a single model:
```bash
$ source eval_models.sh checkpoint/samantha/fasp data/samantha/dataset.pckl
```

To eval all models:
```bash
$ source eval_models.sh checkpoint/samantha/ data/samantha/dataset.pckl
```
