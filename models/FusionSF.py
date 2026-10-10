"""Original FusionSF backbones behind the common TSLib forward signature."""

from dataclasses import asdict

import torch
from torch import nn

from layers.fusionsf.fusionSF_2modal import FusionSF2M
from layers.fusionsf.fusionSF_3modal import FusionSF3M
from layers.fusionsf.layers.positional_encoding import Cyclical_embedding
from .configs import FusionSFConfig


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.seq_len != configs.pred_len:
            raise ValueError('Original FusionSF requires seq_len == pred_len')
        self.three_modal = configs.fusion_modalities == 3
        self.guide_channels = configs.guide_channels
        architecture = FusionSFConfig(input_len=configs.seq_len, output_len=configs.pred_len,
                                      mlp_ratio=4 if self.three_modal else 1,
                                      guide_channels=self.guide_channels)
        if configs.preset == 'script':
            if not self.three_modal:
                raise ValueError('FusionSF script preset requires three modalities')
            architecture.ctx_masking_ratio = 0.99
            architecture.vq_in_ts = architecture.vq_in_ctx = True
        params = asdict(architecture)
        params.pop('input_len')
        params.pop('output_len')
        frequencies = params.pop('frequencies')
        if not self.three_modal:
            for key in ('guide_channels', 'vq_in_ts', 'vq_in_ctx', 'vq_in_guide'):
                params.pop(key)
        backbone = FusionSF3M if self.three_modal else FusionSF2M
        self.backbone = backbone(**params, ts_length=configs.seq_len,
                                 time_coords_encoder=Cyclical_embedding(frequencies=frequencies))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        cov = covariates
        values = [cov['satellite'], cov['satellite_coords'], x_enc,
                  cov['site_coords'][..., None, None], cov['fusion_time']]
        if self.three_modal:
            nwp = cov['future_nwp']
            if self.guide_channels == 17:
                # The author NWP CSV retains lat/lon as its first two columns.
                coords = (torch.round(cov['site_coords'] * 10) / 10).unsqueeze(1)
                nwp = torch.cat([coords.expand(-1, nwp.size(1), -1), nwp], dim=-1)
            values.append(nwp)
        result = self.backbone(*values, mask=self.training)
        if self.three_modal and self.training:
            prediction, vq_loss = result
        else:
            prediction, vq_loss = result, x_enc.new_zeros(())
        return {'prediction': prediction.mean(dim=2), 'vq_loss': vq_loss.mean()}
