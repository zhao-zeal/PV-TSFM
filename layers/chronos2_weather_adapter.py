"""OIB-inspired hourly power residual; native Chronos-2 outputs stay unchanged."""

import torch
from torch import nn


class HorizonWeatherResidualAdapter(nn.Module):
    def __init__(self, model_dim=768, weather_dim=15, hidden_dim=32, patch_size=16):
        super().__init__()
        self.weather_dim = weather_dim
        self.patch_size = patch_size
        self.hidden_state_out = nn.Sequential(nn.LayerNorm(model_dim),
                                              nn.Linear(model_dim, hidden_dim))
        self.cov_out = nn.Linear(weather_dim * 2, hidden_dim)
        self.concat_layer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                          nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.concat_layer[-1].weight)
        nn.init.zeros_(self.concat_layer[-1].bias)

    def forward(self, forecast_embeds, future_nwp):
        if future_nwp.ndim != 3 or future_nwp.shape[-1] != self.weather_dim:
            raise ValueError('Weather Adapter requires [batch, horizon, 15] future NWP')
        if forecast_embeds.ndim != 3 or forecast_embeds.size(0) != future_nwp.size(0):
            raise ValueError('Forecast embeddings and NWP batch sizes must match')
        horizon = future_nwp.size(1)
        if (horizon + self.patch_size - 1) // self.patch_size != forecast_embeds.size(1):
            raise ValueError('Forecast patch count does not match the NWP horizon')
        if torch.isinf(future_nwp).any():
            raise ValueError('NWP contains infinity')
        # 24 hours: patch 0 -> hours 0..15; patch 1 -> hours 16..23.
        hidden = self.hidden_state_out(forecast_embeds).repeat_interleave(
            self.patch_size, dim=1)[:, :horizon]
        missing = future_nwp.isnan()
        weather = torch.cat([future_nwp.masked_fill(missing, 0.),
                             missing.to(future_nwp.dtype)], dim=-1)
        joined = torch.relu(torch.cat([hidden, self.cov_out(weather)], dim=-1))
        return self.concat_layer(joined)
