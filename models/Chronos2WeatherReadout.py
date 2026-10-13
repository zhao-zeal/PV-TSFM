"""A04 preserves accepted A01 encoding, forecast and checkpoint provenance."""

import torch

from layers.chronos2_weather_readout import WeatherConditionedReadout
from models.Chronos2WeatherAdapter import Model as WeatherAdapterModel


class Model(WeatherAdapterModel):
    def __init__(self, configs):
        super().__init__(configs)
        self.memory_variant = configs.memory_variant
        if self.memory_variant not in ('a', 'b'):
            raise ValueError('A04 memory variant must be a or b')
        previous = self.weather_adapter
        # Same A02 initialization and RNG stream, without an additional random draw.
        with torch.random.fork_rng(devices=[]):
            readout = WeatherConditionedReadout(
                model_dim=self.base.backbone.config.d_model, weather_dim=previous.weather_dim,
                hidden_dim=previous.cov_out.out_features, patch_size=previous.patch_size)
        readout.load_state_dict(previous.state_dict(), strict=True)
        self.weather_adapter = readout

    @torch.no_grad()
    def encode_memory(self, power, history, future):
        """Identical packing/encode call to A01; additionally retain context positions."""
        batch, length = power.shape
        group_size = 1 + history.size(-1)
        context = torch.cat((power.unsqueeze(1), history.transpose(1, 2)), 1)
        unknown_power = torch.full_like(future[:, :, :1].transpose(1, 2), torch.nan)
        future_group = torch.cat((unknown_power, future.transpose(1, 2)), 1)
        # Match the accepted A01 call, whose forecast count is fixed at two.
        patch_count = 2
        encoded, (loc, scale), _, context_count = self.base.backbone.encode(
            context=context.reshape(batch * group_size, length),
            future_covariates=future_group.reshape(batch * group_size, self.base.config.pred_len),
            group_ids=torch.arange(batch, device=power.device).repeat_interleave(group_size),
            num_output_patches=patch_count)
        indices = torch.arange(batch, device=power.device) * group_size
        values = encoded[0][indices]
        # Exclude REG between context and forecast positions.
        return values[:, -patch_count:], loc[indices], scale[indices], values[:, :context_count]

    def forward_features(self, embeds, loc, scale, future_nwp, context=None, **options):
        with torch.no_grad():
            adapted = self.base.adapter(embeds)
            forecast = self.base.forward_features(embeds, loc, scale)
            if self.memory_variant == 'b':
                if context is None or context.ndim != 3 or context.size(0) != embeds.size(0):
                    raise ValueError('A04-b requires unchanged contextual power tokens')
                memory = torch.cat((context, adapted), 1)
            else:
                memory = adapted
        result = self.weather_adapter(memory, future_nwp, **options)
        if options.get('return_attention'):
            residual, attention = result
            return forecast + residual, attention
        return forecast + result

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        embeds, loc, scale, context = self.encode_memory(
            x_enc[..., 0], covariates['historical_nwp'], covariates['future_nwp'])
        return self.forward_features(embeds, loc, scale, covariates['future_nwp'], context)
