# Benchmark and analysis of organoids

This is just simple documentation, to change later for something much cleaner.

### Handling the data
```python
import bao
# download and prepare the dataset
bao.datasets.prepare_dataset(dataset_name='orgasegment', should_download=True)
# check for the splits used
train_names, val_names, test_names = bao.datasets.get_dataset_split(dataset_name='orgasegment', split=0, num_splits=4)
```
### Training a model
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
model = WrapperModel(torchvision.models.segmentation.deeplabv3_resnet50())
# train the model
bao.train(model, dataset_name='orgasegment', split=0)
```

### Evaluating a model
```python
import bao
import torch
import torchvision
# prepare the model to evaluate
class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def forward(self, x):
        return torch.nn.functional.sigmoid(self.model(x)['out'])
model = WrapperModel(torchvision.models.segmentation.deeplabv3_resnet50())
# evaluate the model
bao.eval(model, dataset_name='orgasegment', split=0, checkpoint_path='/path/to/checkpoint')
```