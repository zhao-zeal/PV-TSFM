"""BasicTS interface; the original FusionSF backbones remain unchanged."""

from torch import nn
from torch.nn import functional as F

from .fusionSF_2modal import FusionSF2M
from .fusionSF_3modal import FusionSF3M
from .layers.positional_encoding import Cyclical_embedding


def build_backbone(config, backbone_type):
    params = dict(config)
    input_len = params.pop('input_len')
    output_len = params.pop('output_len')
    if input_len != output_len:
        raise ValueError('Original FusionSF requires input_len == output_len')
    frequencies = params.pop('frequencies')
    if backbone_type is FusionSF2M:
        for key in ('guide_channels', 'vq_in_ts', 'vq_in_ctx', 'vq_in_guide'):
            params.pop(key)
    return backbone_type(
        **params, ts_length=input_len,
        time_coords_encoder=Cyclical_embedding(frequencies=frequencies),
    )


class FusionSF2Modal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config, FusionSF2M)

    def forward(self, inputs, stl_input, stl_coords, ts_coords, ts_time):
        prediction = self.backbone(
            stl_input.float(), stl_coords.float(), inputs.float(),
            ts_coords.float(), ts_time.float(), mask=self.training,
        )
        return {'prediction': prediction.mean(dim=2), 'vq_loss': inputs.new_zeros(())}


class FusionSF3Modal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config, FusionSF3M)

    def forward(self, inputs, stl_input, stl_coords, ts_coords, ts_time, ec_input):
        result = self.backbone(
            stl_input.float(), stl_coords.float(), inputs.float(),
            ts_coords.float(), ts_time.float(), ec_input.float(), mask=self.training,
        )
        if self.training:
            prediction, vq_loss = result
        else:
            prediction, vq_loss = result, inputs.new_zeros(())
        return {'prediction': prediction.mean(dim=2), 'vq_loss': vq_loss.mean()}


def fusionsf_loss(prediction, targets, vq_loss):
    """Original L1 regression loss plus the training-only VQ loss."""
    loss = F.l1_loss(prediction, targets)
    return loss + vq_loss
