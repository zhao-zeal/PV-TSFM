"""FusionSF baseline implementations with BasicTS's prediction interface."""

from dataclasses import dataclass

import torch
from torch import nn

from basicts.configs import BasicTSModelConfig

from .dlinear import Model as DLinearModel
from .patchtst import Model as PatchTSTModel


@dataclass
class DLinearConfig(BasicTSModelConfig):
    input_len: int = 24
    output_len: int = 24
    enc_in: int = 3
    moving_avg: int = 13
    individual: bool = False


@dataclass
class PatchTSTConfig(BasicTSModelConfig):
    input_len: int = 24
    output_len: int = 24
    enc_in: int = 3
    c_out: int = 3
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 2048
    factor: int = 5
    dropout: float = 0.05
    embed: str = 'timeF'
    activation: str = 'gelu'
    output_attention: bool = False
    patch_len: int = 16
    stride: int = 8


def build_model(config, model_type):
    params = dict(config)
    params['seq_len'] = params.pop('input_len')
    params['pred_len'] = params.pop('output_len')
    return model_type(**params)


class FusionSFDLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_model(config, DLinearModel)

    def forward(self, inputs, ts_coords):
        coords = ts_coords.float().unsqueeze(1).expand(-1, inputs.shape[1], -1)
        prediction = self.backbone(torch.cat([inputs, coords], dim=-1), None, None, None)
        return prediction[..., :1]


class FusionSFPatchTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_model(config, PatchTSTModel)

    def forward(self, inputs, ts_coords):
        coords = ts_coords.float().unsqueeze(1).expand(-1, inputs.shape[1], -1)
        prediction = self.backbone(torch.cat([inputs, coords], dim=-1), None, None, None)
        return prediction[..., :1]
