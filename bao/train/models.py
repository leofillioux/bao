import os
import torch
import torchmetrics
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
            self.metric = torchmetrics.F1Score(task='binary')
            self.model = nn.Sequential(self.model, nn.LazyLinear(1))
        else:
            self.metric = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
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
        metric = self.metric(self.train_preds, self.train_labels)
        self.log('train/metric', metric, on_epoch=True, on_step=False, prog_bar=True)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_start(self):
        self.val_labels = []
        self.val_preds = []
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        self.val_labels = torch.cat(self.val_labels)
        self.val_preds = torch.cat(self.val_preds)
        metric = self.metric(self.val_preds, self.val_labels)
        self.log('val/metric', metric, on_epoch=True, on_step=False, prog_bar=True)
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
        self.log('train/metric', metric, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)

        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val/metric', metric, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_fit_end(self):
        self.__class__.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path)
        self.model.eval()

        for images, masks in self.trainer.val_dataloaders:
            for idx in range(len(images)):
                image, mask = images[idx], masks[idx]
                img_tensor = image.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred_mask = self(img_tensor)
                pred_mask = pred_mask.squeeze().cpu().numpy() > 0.5
                true_mask = mask.squeeze().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow((image.permute(1, 2, 0).cpu().numpy()*255.).astype(int))
                axs[0].set_title('Input Image')
                axs[0].axis('off')
                axs[1].imshow(true_mask, cmap='gray')
                axs[1].set_title('True Mask')
                axs[1].axis('off')
                axs[2].imshow(pred_mask, cmap='gray')
                axs[2].set_title('Predicted Mask')
                axs[2].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.trainer.checkpoint_callback.best_model_path.rsplit('/', 1)[0], f'val_{idx}.png'))
                plt.show()
            break
    
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
        self.model.train()
        with torch.no_grad():
            loss = self(x, y)
            loss = sum([loss[k] for k in loss])
        self.model.eval()

        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(x))
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