import torch
from dinov2.models import vits
from huggingface_hub import hf_hub_download

def core_base(pretrained: bool = True):
    arch = 'vit_base'
    vit_kwargs = dict(
        img_size=224,
        patch_size=14,
        init_values=1e-05,
        ffn_layer='swiglufused',
        block_chunks=1,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=0,
        interpolate_offset=0.1,
        interpolate_antialias=False,
    )
    core = vits.__dict__[arch](**vit_kwargs)
    if pretrained:
        checkpoint = torch.load(hf_hub_download('leoflx/core', 'core_base.pth'))
        core.load_state_dict(checkpoint)
    return core