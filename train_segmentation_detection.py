import bao
import timm
import torch
import argparse
import torchvision
from copy import copy

segmentation_datasets = ['orgasegment', 'orgaextractor']
detection_datasets = ['orgaquant', 'tellu', 'multiorg']

parser = argparse.ArgumentParser('Train a Segmenter/DETR head on top of the features from foundation models.')
# parameters for both
parser.add_argument('-d',
                    '--dataset',
                    type=str,
                    help='Dataset on which to train.',
                    choices=segmentation_datasets+detection_datasets)
parser.add_argument('-m',
                    '--model',
                    type=str,
                    help='Foundation model to use for the feature extraction.',
                    choices=['vit', 'mae', 'clip', 'dinov2', 'core'])
parser.add_argument('--num_patches',
                    help='The number of patches into which to divide each axis of the image.',
                    type=int)
parser.add_argument('-s',
                    '--seeds',
                    type=int,
                    help='Seeds across which to train the model.',                                                                                                                                              nargs='+',
                    default=list(range(5)))
# parameters for detection
parser.add_argument('--num_queries',
                    help='Maximum number of queries for the DETR prediction head.',
                    type=int)
parser.add_argument('--num_decoder_heads',
                    help='Number of heads for the Transformer decoder.',
                    type=int,
                    default=8)
parser.add_argument('--num_decoder_layers',
                    help='Number of Transformer decoder layers.',
                    type=int,
                    default=6)
parser.add_argument('--cost_giou',
                    help='Weight for the GIoU part of the cost function.',
                    type=int,
                    default=2)
parser.add_argument('--eos_coef',
                    help='Weigth given to the no-object category.',
                    type=float,
                    default=0.1)
parser.add_argument('--cost_class',
                    help='Weight for the class part of the cost function.',
                    type=int,
                    default=1)
parser.add_argument('--cost_bbox',
                    help='Weight for the bounding box part of the cost function.',
                    type=int,
                    default=5)
args = parser.parse_args()

def vit_mae_hook_fn(module, input, output, n_cls=1):
    global full_sequence
    full_sequence.append(output[:, n_cls:, :].detach().cpu())

if args.model == 'vit':
    model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
    model.heads = torch.nn.Identity()
    model.encoder.ln.register_forward_hook(vit_mae_hook_fn)
    model.num_features = 768
elif args.model == 'mae':
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
    model.norm.register_forward_hook(vit_mae_hook_fn)
    model.num_features = 768
elif args.model == 'clip':
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').vision_model
    model.num_features = 768
elif args.model == 'dinov2':
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.num_features = 768
elif args.model == 'core':
    import torch
    model = torch.hub.load('leofillioux/core', 'core_base')
    model.num_features = 768

if args.model in ['vit', 'mae']:
    def forward(model, x, device):
        global full_sequence
        full_sequence = []
        with torch.no_grad():
            model(x.to(device))
        return full_sequence[0]
elif args.model == 'clip':
    def forward(model, x, device):
        with torch.no_grad():
            return model(x).last_hidden_state[..., 1:, :]
elif args.model in ['dinov2', 'core']:
    def forward(model, x, device):
        with torch.no_grad():
            pred = model(x.to(device), is_training=True)['x_norm_patchtokens'].detach().cpu()
        return pred
model.forward_pass = forward

if args.dataset in segmentation_datasets:
    all_metrics = []
    for seed in args.seeds:
        for split in range(4):
            metrics = bao.train(copy(model),
                                args.dataset,
                                split=split,
                                download=False,
                                precomputed=False,
                                seed=seed,
                                batch_size=4,
                                max_epochs=1000,
                                optimizer_params={'lr': 0.00001},
                                task_specific={'segmenter_config': bao.configs.train_config.segmenter_config[args.model],
                                              'img_size': bao.configs.dataset_config[args.dataset].size[0],
                                              'num_patches': args.num_patches})
            dice = metrics["val/metric_dice"].item()
            all_metrics.append(dice)
            print(f'{args.model} seed {seed} split {split} Dice {dice}')
    print(f'{args.model} average Dice {sum(all_metrics) / len(all_metrics)}')
else:
    all_metrics = []
    for seed in args.seeds:
        for split in range(4):
            metrics = bao.train(copy(model),
                                args.dataset,
                                split=split,
                                download=False,
                                precomputed=False,
                                seed=seed,
                                batch_size=4,
                                max_epochs=1000,
                                optimizer_params={'lr': 0.00001},
                                task_specific={'num_patches': args.num_patches,
                                               'num_queries': args.num_queries,
                                               'dataset': args.dataset,
                                               'cost_giou': args.cost_giou,
                                               'eos_coef': args.eos_coef,
                                               'cost_class': args.cost_class,
                                               'cost_bbox': args.cost_bbox,
                                               'img_size': bao.configs.dataset_config[args.dataset].size[0],
                                               'num_decoder_heads': args.num_decoder_heads,
                                               'num_decoder_layers': args.num_decoder_layers})
            map50 = metrics["val/metric_map50"].item()
            all_metrics.append(map50)
            print(f'{args.model} seed {seed} split {split} MaP@50 {map50}')
    print(f'{args.model} average MaP@50 {sum(all_metrics) / len(all_metrics)}')