import bao
import argparse
from copy import copy

parser = argparse.ArgumentParser('Train a linear probing head on top of the features from foundation models.')
parser.add_argument('-d',
                    '--dataset',
                    type=str,
                    help='Dataset on which to train.',
                    choices=['dcrypto_budding', 'dcrypto_opaque'])
parser.add_argument('-m',
                    '--model',
                    type=str,
                    help='Foundation model to use for the feature extraction.',
                    choices=['resnet', 'convnext', 'vit', 'swin', 'mae', 'vicreg', 'clip', 'dinov2'])
args = parser.parse_args()

if args.model == 'resnet':
    import torch
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.fc = torch.nn.Identity()
elif args.model == 'convnext':
    import torch
    import torchvision
    model = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
    model.classifier[-1] = torch.nn.Identity()
elif args.model == 'vit':
    import torch
    import torchvision
    model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
    model.heads = torch.nn.Identity()
elif args.model == 'swin':
    import torch
    import torchvision
    model = torchvision.models.swin_b(weights='IMAGENET1K_V1')
    model.head = torch.nn.Identity()
elif args.model == 'mae':
    import timm
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
elif args.model == 'vicreg':
    import torch
    model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50x2')
# elif args.model == 'clip':
#     import transformers
#     model = transformers.CLIPModel.from_pretrained(model_path).vision_model
elif args.model == 'dinov2':
    import torch
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
else:
    raise NotImplementedError

all_metrics = []
for seed in range(5):
    for split in range(4):
        metrics = bao.train(copy(model),
                            args.dataset,
                            split=split,
                            download=False,
                            precomputed=True,
                            seed=seed,
                            batch_size=64,
                            max_epochs=10000,
                            optimizer_params={'lr': 0.001})
        f1 = metrics["val/metric_f1"].item()
        all_metrics.append(f1)
        print(f'{args.model} seed {seed} split {split} F1 {f1}')
print(f'{args.model} average F1 {sum(all_metrics) / len(all_metrics)}')