"""Weather queries read shared key/value memory with A02's exact parameters."""

import math

import torch

from layers.chronos2_weather_adapter import HorizonWeatherResidualAdapter


class WeatherConditionedReadout(HorizonWeatherResidualAdapter):
    def forward(self, memory, future_nwp, *, query_weather=None, uniform=False,
                return_attention=False):
        if memory.ndim != 3 or future_nwp.ndim != 3 or memory.size(0) != future_nwp.size(0):
            raise ValueError('Expected memory [B,M,D] and future NWP [B,H,C]')
        if future_nwp.size(-1) != self.weather_dim or torch.isinf(future_nwp).any():
            raise ValueError('Invalid future NWP channels or infinity')
        missing = future_nwp.isnan()
        weather = self.cov_out(torch.cat((future_nwp.masked_fill(missing, 0),
                                        missing.to(future_nwp.dtype)), -1))
        keys_values = self.hidden_state_out(memory)
        width = keys_values.size(-1)
        # Fixed position encoding changes Query only; direct weather branch is held intact.
        position = torch.arange(future_nwp.size(1), device=memory.device, dtype=memory.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, width, 2, device=memory.device, dtype=memory.dtype)
                              * (-math.log(10000.) / width))
        encoding = torch.zeros(future_nwp.size(1), width, device=memory.device, dtype=memory.dtype)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[:width // 2])
        query = (weather if query_weather is None else query_weather) + encoding
        attention = torch.softmax(query @ keys_values.transpose(-1, -2) / math.sqrt(width), -1)
        if uniform:
            attention = torch.ones_like(attention) / memory.size(1)
        retrieved = attention @ keys_values
        residual = self.concat_layer(torch.relu(torch.cat((retrieved, weather), -1)))
        return (residual, attention) if return_attention else residual
