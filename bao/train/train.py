import os
import torch
import lightning as L
from .models import get_model
from .datasets import get_dataset
from torch.utils.data import DataLoader
from ..datasets import get_dataset_split
from ..configs import train_config, dataset_config, default_data_dir
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

def train(model: torch.nn.Module,
          dataset_name: str,
          split: int,
          data_dir: str = default_data_dir,
          num_splits: int = 4) -> None:
    """
        TODO
    """
    L.seed_everything(train_config.seed, workers=True)

    assert dataset_name in dataset_config, f'Dataset {dataset_name} not found in dataset_configs.yaml.'

    model = get_model(dataset_config[dataset_name].task)(model,
                                                         train_config.optimizer,
                                                         train_config.optimizer_params,
                                                         dataset_config[dataset_name].num_classes)
    
    img_size = dataset_config[dataset_name].size
    train_names, val_names, test_names = get_dataset_split(dataset_name=dataset_name,
                                                           split=split,
                                                           num_splits=num_splits,
                                                           data_dir=data_dir)

    train_dataset = get_dataset(dataset_config[dataset_name].task)(dataset_name=dataset_name,
                                                                   data_dir=data_dir,
                                                                   names=train_names,
                                                                   mode='train',
                                                                   img_size=img_size)
    val_dataset = get_dataset(dataset_config[dataset_name].task)(dataset_name=dataset_name,
                                                                 data_dir=data_dir,
                                                                 names=val_names,
                                                                 mode='train',
                                                                 img_size=img_size)
    test_dataset = get_dataset(dataset_config[dataset_name].task)(dataset_name=dataset_name,
                                                                  data_dir=data_dir,
                                                                  names=test_names,
                                                                  mode='test',
                                                                  img_size=img_size) if test_names is not None else None
        
    collate_fn = getattr(model, 'collate_fn', None)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config.batch_size,
                                  shuffle=True,
                                  num_workers=train_config.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=train_config.batch_size,
                                shuffle=False,
                                num_workers=train_config.num_workers,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=train_config.batch_size,
                                 shuffle=False,
                                 num_workers=train_config.num_workers,
                                 collate_fn=collate_fn) if test_names is not None else None
    
    early_stopping = EarlyStopping(monitor='val/loss',
                                   patience=train_config.patience)
    model_checkpoint = ModelCheckpoint(monitor='val/loss', mode='min', save_last=False, save_top_k=1)
    callbacks = [early_stopping, model_checkpoint]
    trainer = L.Trainer(max_epochs=train_config.max_epochs,
                        callbacks=callbacks,
                        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                        devices=1,
                        log_every_n_steps=1)
    trainer.fit(model,
                train_dataloader,
                val_dataloader)

    trainer.validate(model,
                     test_dataloader if test_names is not None else val_dataloader,
                     ckpt_path=model_checkpoint.best_model_path)
