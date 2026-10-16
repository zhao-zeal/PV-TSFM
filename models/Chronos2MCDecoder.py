"""Frozen accepted A01 plus the A05 P-Corr/MC Decoder residual head."""

import torch

from layers.chronos2_mc_decoder import A05Head, official_pcorr_features
from layers.chronos2_token_adapter import patch_metadata
from models.Chronos2WeatherAdapter import Model as FrozenA01Model


class Model(FrozenA01Model):
    def __init__(self, configs):
        super().__init__(configs)
        # The parent is reused only for its accepted A01 loading/provenance checks.
        # A05 does not include A02's weather residual module.
        del self.weather_adapter
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(configs.seed)
            self.a05_head = A05Head(
                configs.a05_variant, configs.a05_width, configs.a05_heads,
                configs.a05_ff, configs.dropout)

    @torch.no_grad()
    def encode_all(self, power, historical_nwp, future_nwp, correlation_history):
        batch, length, _ = power.shape
        channels = historical_nwp.size(-1) + 1
        context = torch.cat((power, historical_nwp), -1).transpose(1, 2)
        future = torch.cat((torch.full_like(future_nwp[..., :1], torch.nan), future_nwp), -1).transpose(1, 2)
        encoded, (loc, scale), future_mask, n_history = self.base.backbone.encode(
            context=context.reshape(batch * channels, length),
            future_covariates=future.reshape(batch * channels, future_nwp.size(1)),
            group_ids=torch.arange(batch, device=power.device).repeat_interleave(channels),
            num_output_patches=2)
        states = encoded[0].reshape(batch, channels, -1, 768)
        if n_history != 2 or states.size(2) != 5:
            raise ValueError(f'A05 24->24 expects H0,H1,REG,F0,F1; got {states.size(2)} states')
        tokens = torch.cat((states[:, :, :n_history], states[:, :, -2:]), 2)
        target_indices = torch.arange(batch, device=power.device) * channels
        base = self.base.forward_features(states[:, 0, -2:], loc[target_indices], scale[target_indices])
        if future_mask[target_indices].bool().any():
            raise ValueError('Chronos target future must remain unobserved')
        metadata, positions = patch_metadata(power, historical_nwp, future_nwp)
        pcorr = official_pcorr_features(power, historical_nwp, correlation_history)
        return tokens, base, metadata, pcorr, positions

    def forward_cached(self, tokens, base, metadata, pcorr_features, positions):
        return base + self.a05_head(tokens, metadata, positions, pcorr_features)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        values = self.encode_all(
            x_enc, covariates['historical_nwp'], covariates['future_nwp'],
            covariates['correlation_history'])
        return self.forward_cached(*values)
