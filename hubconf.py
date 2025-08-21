import json
import torch
from dinov2.models import vits
from huggingface_hub import hf_hub_download

def core_base(pretrained: bool = True):
    arch = 'vit_base'
    with open(hf_hub_download('leoflx/core', 'config.json'), 'r') as f:
        vit_kwargs = json.load(f)
    core = vits.__dict__[arch](**vit_kwargs)
    if pretrained:
        checkpoint = torch.load(hf_hub_download('leoflx/core', 'core_base.pth'))
        core.load_state_dict(checkpoint)
    return core

model = core_base()
print(model)