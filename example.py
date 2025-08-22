import bao
import torch

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.num_features = 384

# classification
model.forward_pass = lambda model, x: model(x)
dataset = 'dcrypto_budding'
metrics = bao.train(model,
                    dataset,
                    split=0, 
                    seed=0,
                    download=False,
                    batch_size=64,
                    max_epochs=5,
                    task_specific={'precomputed': True},
                    optimizer_params={'lr': 0.001})

# segmentation
def forward_pass(model, x, device):
    # for segmentation and detection you need to return the full sequence
    with torch.no_grad():
        return model(x.to(device), is_training=True)['x_norm_patchtokens'].detach().cpu()
model.forward_pass = forward_pass
dataset = 'orgaextractor'
segmenter_config = {'n_cls': 1,
                    'patch_size': 14,
                    'd_encoder': 384,
                    'n_layers': 2,
                    'n_heads': 3,
                    'd_model': 384,
                    'd_ff': 1536,
                    'drop_path_rate': 0.0,
                    'dropout': 0.0}
metrics = bao.train(model,
                    dataset,
                    split=0,
                    seed=0,
                    download=False,
                    batch_size=4,
                    max_epochs=5,
                    optimizer_params={'lr': 0.00001},
                    task_specific={'segmenter_config': segmenter_config,
                                   'img_size': bao.configs.dataset_config[dataset].size[0],
                                   'num_patches': 14})

# detection
# the same forward pass is used for detection as for segmentation
dataset = 'orgaquant'
metrics = bao.train(model,
                    dataset,
                    split=0,
                    seed=0,
                    download=False,
                    batch_size=4,
                    max_epochs=5,
                    optimizer_params={'lr': 0.00001},
                    task_specific={'num_patches': 1,
                                    'num_queries': 50,
                                    'dataset': dataset,
                                    'cost_giou': 2,
                                    'eos_coef': 0.1,
                                    'cost_class': 1,
                                    'cost_bbox': 5,
                                    'img_size': bao.configs.dataset_config[dataset].size[0],
                                    'num_decoder_heads': 8,
                                    'num_decoder_layers': 6})