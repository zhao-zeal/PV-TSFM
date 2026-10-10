"""Published Cross-Unet with the shared data interface."""

from dataclasses import asdict
from types import SimpleNamespace

from layers.crossunet.official_model import Model as Backbone
from .configs import CrossUnetConfig


class Model(Backbone):
    def __init__(self, configs):
        if configs.seq_len != configs.pred_len:
            raise ValueError('Migrated MMSP Cross-Unet requires seq_len == pred_len')
        params = asdict(CrossUnetConfig(
            input_len=configs.seq_len, output_len=configs.pred_len,
            seg_len=configs.seg_len, history_features=1, weather_features=15,
            d_model=configs.d_model, d_ff=configs.d_ff, e_layers=configs.e_layers,
            n_heads=configs.n_heads, factor=configs.factor, dropout=configs.dropout))
        params.pop('power_only')
        params['enc_in'] = params.pop('history_features') + params.pop('weather_features')
        params['seq_len'] = params.pop('input_len')
        params['pred_len'] = params.pop('output_len')
        super().__init__(SimpleNamespace(**params))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        return super().forward(x_enc, None, covariates['future_nwp'], None,
                               covariates['historical_nwp'], covariates['correlation_history'])[..., -1:]
