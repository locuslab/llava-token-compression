import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, tower_config, delay_load=False, **kwargs):
    modules = []

    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
        modules = build_token_compressor(modules, config, tower_config)
        return nn.Sequential(*modules)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        modules = build_token_compressor(modules, config, tower_config)
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_token_compressor(modules, config, tower_config):
    token_compression_type = getattr(config, 'mm_vision_token_compression_type', None)
    if token_compression_type is not None:
        if token_compression_type == "quecc":
            from .quecc import QueCC
        
        token_compressor = QueCC(tower_config=tower_config, training_config=config)
        token_compressor.requires_grad_(True)

        modules.insert(0, token_compressor)
    
    return modules
