import os
import torch
import torchmetrics
import lightning as L
import torch.nn as nn
from torch import optim
from . import train_utils

def get_model(task: str) -> L.LightningModule:
    """
        Get the model based on the task.
        arguments:
            task [str]: the task to perform
        returns:
            model [L.LightningModule]: the model to use
    """
    if task == 'classification':
        model = ClassificationModel
    elif task == 'segmentation':
        model = SegmentationModel
    elif task in ['detection', 'localization']:
        model = DetectionModel
    else:
        raise ValueError(f'Task {task} not supported.')
    return model

class ClassificationModel(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_name: str,
                 optimizer_params: dict,
                 num_classes: int = 2):
        super(ClassificationModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self._optimizer_name = optimizer_name
        self._optimizer_params = optimizer_params

        self.loss = torch.nn.BCELoss()
        if num_classes == 2:
            self.metric_f1 = torchmetrics.F1Score(task='binary')
            self.metric_auc = torchmetrics.AUROC(task='binary')
            self.model = nn.Sequential(self.model, nn.LazyLinear(1))
        else:
            self.metric_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
            self.metric_auc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes)
            self.model = nn.Sequential(self.model, nn.LazyLinear(num_classes))

    def on_train_epoch_start(self):
        self.train_labels = []
        self.train_preds = []
        return super().on_train_epoch_start()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_labels.append(y)
        self.train_preds.append(y_hat.detach())

        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.train_labels = torch.cat(self.train_labels)
        self.train_preds = torch.cat(self.train_preds)
        metric_f1 = self.metric_f1(self.train_preds, self.train_labels)
        metric_auc = self.metric_auc(self.train_preds, self.train_labels)
        self.log('train/metric_f1', metric_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train/metric_auc', metric_auc, on_epoch=True, on_step=False, prog_bar=True)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_start(self):
        self.val_labels = []
        self.val_preds = []
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        self.val_labels = torch.cat(self.val_labels)
        self.val_preds = torch.cat(self.val_preds)
        metric_f1 = self.metric_f1(self.val_preds, self.val_labels)
        metric_auc = self.metric_auc(self.val_preds, self.val_labels)
        self.log('val/metric_f1', metric_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val/metric_auc', metric_auc, on_epoch=True, on_step=False, prog_bar=True)
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.val_labels.append(y)
        self.val_preds.append(y_hat.detach())

        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        return getattr(optim, self._optimizer_name)(self.parameters(), **self._optimizer_params)

class SegmentationModel(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_name: str,
                 optimizer_params: dict,
                 num_classes: int = None):
        super(SegmentationModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self._optimizer_name = optimizer_name
        self._optimizer_params = optimizer_params

        self.loss = train_utils.dice_loss
        self.metric = train_utils.dice_metric

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)

        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train/metric_dice', metric, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)

        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val/metric_dice', metric, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return getattr(optim, self._optimizer_name)(self.parameters(), **self._optimizer_params)

class DetectionModel(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_name: str,
                 optimizer_params: dict,
                 num_classes: int = None):
        super(DetectionModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.metric = torchmetrics.detection.MeanAveragePrecision(iou_type='bbox')
        self._optimizer_name = optimizer_name
        self._optimizer_params = optimizer_params

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        loss = sum([loss[k] for k in loss])
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(x))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y)
        self.metric.update(y_hat, y)
        metric = self.metric.compute()['map_50']

        self.model.train()
        with torch.no_grad():
            loss = self(x, y)
            loss = sum([loss[k] for k in loss])
        self.model.eval()

        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(x))
        self.log('val/metric_map50', metric, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(x))
        return loss
    
    def forward(self, x, y):
        x = list(image for image in x)
        y = [{k: v for k, v in t.items()} for t in y]
        return self.model(x, y)
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def configure_optimizers(self):
        return getattr(optim, self._optimizer_name)(self.parameters(), **self._optimizer_params)
