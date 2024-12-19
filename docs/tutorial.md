# Benchmark and analysis of organoids

## Handling the data
The `bao.datasets` subpackage handles everything relative to the supported datasets in the Bao package.

The **`bao.datasets.prepare_dataset`** function downloads the specified dataset and preprocesses it in the desired format. The name of the dataset should correspond to the name given in `configs/dataset_config.yaml`.
```python
import bao

bao.datasets.prepare_dataset('tellu')
```

The `bao.datasets.get_dataset_split` function splits the samples into train and validation in a deterministic manner (given the seed), and returns test sample names if there is a predefined test set in the original dataset. This assumes that the dataset has already been downloaded and prepared, and is present in the directory given in data_dir (`data` by default).
```python
import bao

train_names, val_names, test_names = bao.datasets.get_dataset_split(dataset_name='tellu',
                                                                    split=0,
                                                                    num_splits=4)
```

The `bao.datasets.get_dataset` function gives access to the specific dataset classes and initializes it.
```python
import bao

train_names, _, _ = bao.datasets.get_dataset_split(dataset_name='tellu',
                                                   split=0,
                                                   num_splits=4)
train_dataset = bao.datasets.get_dataset(name='tellu',
                                         names=train_names,
                                         download=False, # has already been downloaded above
                                         mode='train', # mode should be train for train and val
                                         img_size=(1024, 1024))
```

All speicific dataset classes are also directly accessible.
```python
import bao

train_names, _, _ = bao.datasets.get_dataset_split(dataset_name='tellu',
                                                   split=0,
                                                   num_splits=4)
train_dataset = bao.datasets.Tellu(names=train_names
                                   mode='train',
                                   img_size=(1024, 1024),
                                   download=False)
```

## Training and evaluating a model
The `bao.train` function is a packaged way of training your model. This can be done either by:
- defining your `torch.nn.Module` model, which should output only the tensor output (i.e., no dictionary)
- defining your `lightning.LightningModule` model, for which you can define your own losses (make sure to log at least 'train/loss' and 'val/loss')

The default hyperparameters for training are all defined in `configs/train_config.yaml`, if you wish to modify or add some hyperparameters, you may do so with the **kwargs.

**Note**: if you prefer to train a model using your own pipeline, you can always use the dataset classes defined above with your own pipeline.

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
bao.train(model, dataset_name='orgasegment', split=0, batch_size=8, optimizer_params={'lr': 0.01, 'weight_decay': 0.001})
```
 The similar code works, providing a Lightning.LightningModule as input.
```python
import bao
import torch
import torchvision
import lightning as L
# prepare the model to train
class CustomLightning(L.LightningModule):
    def __init__(self):
        super(CustomLightning, self).__init__() 
        self.loss = torch.nn.BCELoss()
        self.model = torch.nn.Sequential(torchvision.models.vgg16(), torch.nn.LazyLinear(1), torch.nn.Sigmoid())
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
# train the model
model = CustomLightning()
bao.train(model, dataset_name='dcrypto_budding', split=0, download=True)
```

The `bao.eval` function evaluates a trained model on a specified dataset (which is already done at the end of the training), on the test set if one is present, or on the validation set. Similarly to `bao.train`, it can be applied on a `torch.nn.Module` or on a `lightning.LightningModule`.
```python
import bao

model = WrapperModel(torchvision.models.segmentation.deeplabv3_resnet50(num_classes=1))
bao.eval(model, dataset_name='dcrypto_budding', split=0, download=False, checkpoint_path='/path/to/model.ckpt')
```

## Adding a new dataset to Bao
If a new open-source dataset has come out, feel free to add your implementation and create a pull request. Here are the steps necessary to handle a new dataset.

1. Add the necessary information in `bao/configs/dataset_config.yaml` (download link, name, type of task, image size, number of classes).

2. Create a function called `prepare_<dataset_name>` in `bao/datasets/prepare_utils.py`, which should create a .h5 file with the following structure for the train set (and same for `test` if there is a predefined test set).
    - classification: `train/cls1`, `train/cls2`, ...
    - segmentation: `train/images` and `train/masks`
    - detection: `train/images` and `train/labels` containing ([cls, start_x, start_y, end_x, end_y] arrays for the labels)
    - localization: `train/images` and `train/labels` containing ([start_x, start_y, end_x, end_y] arrays for the labels)
    
3. Create the specific dataset class (e.g. if the dataset is for segmentation, in `bao/datasets/segmentation_datasets.py` which inherits from `SegmentationDataset`), and import it in `bao/configs/__init__.py`.
