"""A03-a/b/c: frozen A01 plus A02 weather residual conditioned on history."""

import torch
from layers.chronos2_history_weather_adapter import HistoricalRelationConditioner
from models.Chronos2WeatherAdapter import Model as WeatherModel


class Model(WeatherModel):
    def __init__(self, configs):
        # Construct A02 first so its initial trainable weights match seed42 A02.
        super().__init__(configs)
        self.history_relation = HistoricalRelationConditioner(
            configs.history_variant, self.weather_adapter.weather_dim,
            self.weather_adapter.cov_out.out_features)

    def forward_features(self, embeds, loc, scale, future_nwp, power, history_nwp):
        adapter = self.weather_adapter
        if (future_nwp.ndim != 3 or future_nwp.size(-1) != adapter.weather_dim
                or future_nwp.size(0) != embeds.size(0)
                or power.size(0) != embeds.size(0)):
            raise ValueError('Forecast features, history and future weather shapes differ')
        horizon = future_nwp.size(1)
        if (horizon + adapter.patch_size - 1) // adapter.patch_size != embeds.size(1):
            raise ValueError('Forecast patch count differs from future horizon')
        if torch.isinf(future_nwp).any():
            raise ValueError('Future NWP contains infinity, as in A02')
        with torch.no_grad():
            adapted = self.base.adapter(embeds)
            forecast = self.base.forward_features(embeds, loc, scale)
        hidden = adapter.hidden_state_out(adapted).repeat_interleave(
            adapter.patch_size, dim=1)[:, :horizon]
        missing = future_nwp.isnan()
        weather = torch.cat((future_nwp.masked_fill(missing, 0),
                             missing.to(future_nwp.dtype)), dim=-1)
        encoded_weather = adapter.cov_out(weather)
        relation = self.history_relation(power, history_nwp)
        # Bounded sample-specific scale in [0,2]; weather is still explicitly encoded.
        conditioned = encoded_weather * (1 + relation.tanh()[:, None])
        joined = torch.relu(torch.cat((hidden, conditioned), dim=-1))
        return forecast + adapter.concat_layer(joined)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        features = self.base.encode_inputs(
            x_enc[..., 0], covariates['historical_nwp'], covariates['future_nwp'])
        return self.forward_features(*features, covariates['future_nwp'],
                                     x_enc[..., :1], covariates['historical_nwp'])
