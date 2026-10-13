"""FusionSF E11/E12 structures; Chronos encoder and output head stay frozen."""

import torch
from torch import nn


class ForecastOnlyBottleneckAdapter(nn.Module):
    def __init__(self, chronos_dim=768, bottleneck_dim=131, alpha_init=0.05):
        super().__init__()
        self.forecast_adapter = nn.Sequential(
            nn.LayerNorm(chronos_dim), nn.Linear(chronos_dim, bottleneck_dim),
            nn.GELU(), nn.Linear(bottleneck_dim, chronos_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, forecast_embeds):
        return forecast_embeds + self.alpha * self.forecast_adapter(forecast_embeds)


def residual_readout(version):
    if version == 'b131_linear_v1':
        module = nn.Linear(1536, 24)
        final = module
    else:
        module = nn.Sequential(nn.Linear(1536, 63), nn.GELU(), nn.Linear(63, 24))
        final = module[-1]
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    return module
