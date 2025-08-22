<p align="center">
  <img src="docs/assets/banner.png" alt="bao logo" width="100%"/>
</p>

# Benchmark and analysis of organoids
This is the official implementation of Baseline and analysis of organoids.

## Documentation
Check [Bao's documentation](https://leofillioux.github.io/bao/) to get started.

## Installation
Bao can be installed using `pip`for `python>=3.10`.
```bash
pip install bao_bench
```

If you want to add new datasets, new tasks, ... you can clone the dataset and install.
```bash
pip install -e .
```

## Using CORE
We release the CORE foundation model which can easily be accessed.
```python
import torch

core = torch.hub.load('leofillioux/bao', 'core_base')
```

## Benchmarking your own model
The [example.py](example.py) script gives an example of how to benchmark your own model across classification, segmentation, and detection tasks.
If you only want to have access to the datasets, you may do so independently of the `bao.train` function. The `bao.datasets` subpackage gives access to an interface to download and prepare datasets, as well as dataset classes.
```python
import bao

train_samples, val_samples, test_samples = bao.datasets.get_dataset_split(dataset_name='tellu', split=0, num_splits=4)
tellu_train_dataset = bao.datasets.Tellu(train_samples, mode='train', img_size=(1024, 1024), download=True)
```

## Reproducing results
The results reported in the paper can be reproduced using [train_classification.py](train_classification.py) and [train_segmentation_detection.py](train_segmentation_detection.py).

```bash
# generic usage for classification
python train_classification.py -d <dataset_name> -m <model_name>
# example use
python train_classification.py -d dcrypto_budding -m core
```

```bash
# generic usage for segmentation
python train_segmentation_detection.py -d <dataset_name> -m <model_name> --num_patches <num_patches>
# example use
python train_segmentation_detection.py -d orgasegment -m core --num_patches 8
```

```bash
# generic usage for segmentation
python train_segmentation_detection.py -d <dataset_name> -m <model_name> --num_patches <num_patches> --num_queries <num_queries>
# example use (only using 2 seeds for MultiOrg)
python train_segmentation_detection.py -d multiorg -m core --num_patches 6 --num_queries 60 -s 0 1
```