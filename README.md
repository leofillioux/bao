<p align="left">
  <img src="docs/assets/logo.png" alt="bao logo" width="20%"/>
</p>

# Benchmark and analysis of organoids
TODO: have a benchmark here, as well as a small description of the datasets.

## Documentation
Check Bao's documentation (TODO put link here) to get started.

## Installation
Bao can be installed using `pip`for `python>=3.10`.
```bash
pip install bao
```

If you want to add new datasets, new tasks, ... you can clone the dataset and install.
```bash
pip install -e .
```

## Usage
The `bao.datasets` subpackage gives access to an interface to download and prepare datasets, as well as dataset classes.
```python
import bao

train_samples, val_samples, test_samples = bao.datasets.get_dataset_split(dataset_name='tellu', split=0, num_splits=4)
tellu_train_dataset = bao.datasets.Tellu(train_samples, mode='train', img_size=(1024, 1024), download=True)
```

The `bao.train` function gives an easy way to train a custom model (either `torch.nn.Module` or `lightning.LightningModule`) on the handled datasets.
```python
import bao
import torch
import torchvision

# prepare the model to train
class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def forward(self, x):
        return torch.nn.functional.sigmoid(self.model(x)['out'])
model = WrapperModel(torchvision.models.segmentation.deeplabv3_resnet50(num_classes=1))

# train the model
bao.train(model, dataset_name='orgasegment', split=0)
```

The `bao.eval` function allows the user to evaluate a trained model on the desired dataset.
```python
import bao

model = WrapperModel(torchvision.models.segmentation.deeplabv3_resnet50(num_classes=1))
bao.eval(model, dataset_name='orgasegment', split=0, checkpoint_path='/path/to/model.ckpt')
```

More tutorials are given here (TODO: make notebooks to show different types of models, and different functionalities).