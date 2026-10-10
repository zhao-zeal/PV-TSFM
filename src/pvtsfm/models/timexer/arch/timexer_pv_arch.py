"""Official MS branch with explicit power and historical NWP inputs."""

from types import SimpleNamespace

import torch

from .official_model import Model


class TimeXerPV(Model):
    def __init__(self, config):
        params = dict(config)
        params['seq_len'] = params.pop('input_len')
        params['pred_len'] = params.pop('output_len')
        params['enc_in'] = params.pop('nwp_channels') + 1
        params.update(features='MS', task_name='long_term_forecast', embed='timeF', freq='h')
        super().__init__(SimpleNamespace(**params))

    def forward(self, inputs, historical_nwp, inputs_timestamps=None):
        """[B,L,1] power + [B,L,C] NWP → [B,H,1] power.

        Optional timestamps are [B,L,4] normalized hourly calendar features.
        Power is last in the official input; future NWP and targets are absent.
        """
        x_enc = torch.cat([historical_nwp, inputs], dim=-1)
        return super().forward(x_enc, inputs_timestamps, None, None)
