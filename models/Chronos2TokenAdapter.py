"""A frozen accepted A01 forecast plus a residual reading native PV/NWP tokens."""

import torch

from layers.chronos2_token_adapter import TokenResidual, history_correlation, patch_metadata
from models.Chronos2WeatherAdapter import Model as WeatherModel


class Model(WeatherModel):
    def __init__(self, configs):
        super().__init__(configs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(configs.seed)
            self.weather_adapter = TokenResidual(configs.token_variant, configs.mlp_width)

    @torch.no_grad()
    def encode_all(self, power, history, future):
        batch, length, _ = power.shape
        channels = history.size(-1) + 1
        context = torch.cat((power, history), -1).transpose(1, 2)
        future_group = torch.cat((torch.full_like(future[..., :1], torch.nan), future), -1).transpose(1, 2)
        encoded, (loc, scale), future_mask, n_history = self.base.backbone.encode(
            context=context.reshape(batch * channels, length),
            future_covariates=future_group.reshape(batch * channels, future.size(1)),
            group_ids=torch.arange(batch, device=power.device).repeat_interleave(channels),
            num_output_patches=2)
        states = encoded[0].reshape(batch, channels, -1, 768)
        tokens = torch.cat((states[:, :, :n_history], states[:, :, -2:]), 2)
        indices = torch.arange(batch, device=power.device) * channels
        # Stage 1 context is exactly the accepted A01 context; future target stays NaN.
        base = self.base.forward_features(states[:, 0, -2:], loc[indices], scale[indices])
        assert not future_mask[indices].bool().any()
        metadata, positions = patch_metadata(power, history, future)
        return tokens, base, metadata, history_correlation(power, history), positions

    def forward_cached(self, tokens, base, future, metadata, correlation, positions):
        return base + self.weather_adapter(tokens, future, correlation, metadata, positions)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        future = covariates['future_nwp']
        tokens, base, metadata, corr, positions = self.encode_all(x_enc, covariates['historical_nwp'], future)
        return self.forward_cached(tokens, base, future, metadata, corr, positions)
