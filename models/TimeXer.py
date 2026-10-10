"""Official MS branch: historical NWP followed by the target power channel."""

import torch
from types import SimpleNamespace

from layers.timexer.official_model import Model as Backbone


class Model(Backbone):
    def __init__(self, configs):
        super().__init__(SimpleNamespace(
            seq_len=configs.seq_len, pred_len=configs.pred_len, enc_in=16,
            features='MS', task_name='long_term_forecast', embed='timeF', freq='h',
            patch_len=configs.patch_len, d_model=configs.d_model, d_ff=configs.d_ff,
            n_heads=configs.n_heads, e_layers=configs.e_layers, factor=configs.factor,
            dropout=configs.dropout, activation='gelu', use_norm=True))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        inputs = torch.cat([covariates['historical_nwp'], x_enc], dim=-1)
        return super().forward(inputs, x_mark_enc, None, None)
