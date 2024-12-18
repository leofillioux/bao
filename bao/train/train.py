from torch.nn import Module
from ..configs import default_data_dir

def train(model: Module,
          dataset_name: str,
          split: int,
          data_dir: str = default_data_dir,
          num_splits: int = 4,
          **kwargs) -> None:
    """Train a model on a specified dataset.
        
        Args:
            model [torch.nn.Module | L.LightningModule]: model to train
            dataset_name [str]: name of the dataset to train on
            split [int]: index of the split to train on
            data_dir [str]: directory to save the prepared dataset to
            num_splits [int]: number of splits to create
            kwargs: additional arguments to update the training configuration
        
        Returns:
            None
    """
    from ..datasets import get_dataset_split
    from ..configs import train_config, dataset_config, default_data_dir
    import torch
    import lightning as L
    from .models import get_model
    from ..datasets import get_dataset
    from torch.utils.data import DataLoader
    from .train_utils import update_easydict
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    updated_train_config = update_easydict(train_config, kwargs)
    L.seed_everything(updated_train_config.seed, workers=True)

    assert dataset_name in dataset_config, f'Dataset {dataset_name} not found in dataset_configs.yaml.'

    if not isinstance(model, L.LightningModule):
        model = get_model(dataset_config[dataset_name].task)(model,
                                                            updated_train_config.optimizer,
                                                            updated_train_config.optimizer_params,
                                                            dataset_config[dataset_name].num_classes)
    
    img_size = dataset_config[dataset_name].size
    train_names, val_names, test_names = get_dataset_split(dataset_name=dataset_name,
                                                           split=split,
                                                           num_splits=num_splits,
                                                           data_dir=data_dir,
                                                           seed=updated_train_config.seed)

    train_dataset = get_dataset(task=dataset_config[dataset_name].task,
                                dataset_name=dataset_name,
                                data_dir=data_dir,
                                names=train_names,
                                mode='train',
                                img_size=img_size)
    val_dataset = get_dataset(dataset_config[dataset_name].task,
                              dataset_name=dataset_name,
                              data_dir=data_dir,
                              names=val_names,
                              mode='train',
                              img_size=img_size)
    test_dataset = get_dataset(dataset_config[dataset_name].task,
                               dataset_name=dataset_name,
                               data_dir=data_dir,
                               names=test_names,
                               mode='test',
                               img_size=img_size) if test_names is not None else None
        
    collate_fn = getattr(model, 'collate_fn', None)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=updated_train_config.batch_size,
                                  shuffle=True,
                                  num_workers=updated_train_config.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=updated_train_config.batch_size,
                                shuffle=False,
                                num_workers=updated_train_config.num_workers,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=updated_train_config.batch_size,
                                 shuffle=False,
                                 num_workers=updated_train_config.num_workers,
                                 collate_fn=collate_fn) if test_names is not None else None
    
    early_stopping = EarlyStopping(monitor='val/loss',
                                   patience=updated_train_config.patience)
    model_checkpoint = ModelCheckpoint(monitor='val/loss', mode='min', save_last=False, save_top_k=1)
    callbacks = [early_stopping, model_checkpoint]
    trainer = L.Trainer(max_epochs=updated_train_config.max_epochs,
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
