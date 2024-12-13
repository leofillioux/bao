# Benchmark and analysis of organoids

## Documentation
### Handling the data
The [**`bao.datasets.prepare_dataset`**](bao/datasets/prepare.py) function downloads the specified dataset and preprocesses it in the desired format. The name of the dataset should correspond to the name given in `configs/dataset_config.yaml`.
| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| dataset_name | str | name of the dataset to prepare |  |
| should_download | bool | whether to download the dataset | True |
| download_dir | str | directory to download the dataset to | '/tmp' |
| data_dir | str | directory to save the prepared dataset to | 'data' |
```python
import bao
bao.datasets.prepare_dataset(dataset_name='orgasegment')
```

The [**`bao.datasets.get_dataset_split`**](bao/datasets/split.py) function splits the dataset into train, validation and testing given the number of splits and the desired split in a deterministic fashion.
| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| dataset_name | str | name of the dataset to prepare | |
| split | int | index of the split to get | |
| num_splits | int | number of splits to create | 4 |
data_dir | str | directory to save the prepared dataset to | 'data' | 
seed | int | seed for the random number generator | 0 |
```python
import bao
train_names, val_names, test_names = bao.datasets.get_dataset_split(dataset_name='orgasegment', split=0, num_splits=4)
```

The [**`bao.datasets.get_dataset`**](bao/datasets/datasets.py) function allows to have access to the dataset classes. Note that the mode given to the dataset should be 'train' for the training and validation set.
| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| task | str | task of the dataset (e.g., classification, segmentation) | |
| data_dir | str | directory to the saved dataset | 'data'
| kwargs | | arguments to pass to the dataset (names, mode, img_size) |
```python
import bao
train_names, _, _ = bao.datasets.get_dataset_split(dataset_name='orgasegment', split=0, num_splits=4)
train_dataset = bao.datasets.get_dataset(task='segmentation', names=train_names, mode='train', img_size=(128, 128))
```

### Training a model
The [**`bao.train`**](bao/train/train.py) function is a packaged way of training your model. This can be done either by:
- defining your `torch.nn.Module` model, which should output only the tensor output (i.e., no dictionary)
- defining your `lightning.LightningModule` model, for which you can define your own losses (make sure to log at least 'train/loss' and 'val/loss', and optionally 'train/metric' and 'val/metric')

The default hyperparameters for training are all defined in `configs/train_config.yaml`, if you wish to modify or add some hyperparameters, you may do so with the **kwargs.

**Note**: if you prefer to train a model using your own pipeline, you can always use the dataset classes defined above with your own pipeline.

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| model | torch.nn.Module or lightning.LightningModule | model to train | |
| dataset_name | str | name of the dataset to train on | |
| split | int | index of the split to train on | |
| data_dir | str | directory to save the prepared dataset to | 'data' |
| num_splits | int | number of splits to create | 4 |
| kwargs | | additional arguments to update the training configuration | |

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

As mentionned above, it is also possible to provide a `lightning.LightningModule` class to the model, in order to customize the training.
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
bao.train(model, dataset_name='dcrypto_budding', split=0)
```

### Evaluating a model
TODO
