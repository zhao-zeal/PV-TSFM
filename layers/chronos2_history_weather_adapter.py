"""Small, independent historical conditioners for the unchanged A02 weather path.

Correlation is descriptive and signed, not a causal channel-importance score.
Missingness means nonfinite input here; upstream interpolated values are observed.
"""

import torch
from torch import nn


def historical_statistics(values):
    """Population mean/std and observed fraction, including all-missing channels."""
    valid = torch.isfinite(values)
    clean = torch.where(valid, values, 0).double()
    count = valid.sum(dim=1)
    mean = clean.sum(dim=1) / count.clamp_min(1)
    centered = torch.where(valid, clean - mean[:, None], 0)
    variance = centered.square().sum(dim=1) / count.clamp_min(1)
    features = torch.cat((mean, variance.clamp_min(0).sqrt(),
                          count / values.size(1)), dim=-1)
    return features.to(values.dtype)


def weather_power_correlation(power, history, variance_epsilon=1e-8):
    """Pairwise-valid Pearson, observation fraction and reliable-pair indicator.

    Fewer than two pairs or either population variance <= epsilon: r=0, valid=0.
    Negative coefficients remain available; no test-dependent thresholds.
    """
    valid = torch.isfinite(power) & torch.isfinite(history)
    p = torch.where(valid, power, 0).double()
    w = torch.where(valid, history, 0).double()
    count = valid.sum(dim=1)
    n = count.clamp_min(1)
    pc = torch.where(valid, p - (p.sum(dim=1) / n)[:, None], 0)
    wc = torch.where(valid, w - (w.sum(dim=1) / n)[:, None], 0)
    pv, wv = pc.square().sum(dim=1) / n, wc.square().sum(dim=1) / n
    reliable = (count >= 2) & (pv > variance_epsilon) & (wv > variance_epsilon)
    denominator = (pv.clamp_min(variance_epsilon) * wv.clamp_min(variance_epsilon)).sqrt()
    correlation = ((pc * wc).sum(dim=1) / n / denominator).clamp(-1, 1)
    correlation = torch.where(reliable, correlation, 0)
    return torch.cat((correlation, count / history.size(1), reliable.to(p.dtype)),
                     dim=-1).to(history.dtype)


class HistoricalRelationConditioner(nn.Module):
    def __init__(self, variant, weather_dim, hidden_dim):
        super().__init__()
        if variant not in ('a', 'b', 'c'):
            raise ValueError('Expected independent A03 variant a, b or c')
        self.variant, self.weather_dim = variant, weather_dim
        # a: 3 statistics per power/weather channel.
        # b: coefficient, observed fraction, reliable indicator per weather channel.
        # c: power/weather, paired products, validity, relative historical position.
        width = 3 * weather_dim if variant == 'b' else 3 * (weather_dim + 1)
        self.encoder = nn.Sequential(nn.Linear(width, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim))

    def features(self, power, history):
        if (power.ndim != 3 or history.ndim != 3 or power.size(-1) != 1
                or power.shape[:2] != history.shape[:2]
                or history.size(-1) != self.weather_dim or history.size(1) < 1):
            raise ValueError('Expected aligned [B,L,1] power and [B,L,C] history NWP')
        if self.variant == 'a':
            return historical_statistics(torch.cat((power, history), dim=-1))
        if self.variant == 'b':
            return weather_power_correlation(power, history)
        values = torch.cat((power, history), dim=-1)
        valid = torch.isfinite(values)
        clean = torch.where(valid, values, 0)
        products = clean[..., :1] * clean[..., 1:]
        position = torch.linspace(-1, 1, history.size(1), device=history.device,
                                  dtype=history.dtype).view(1, -1, 1).expand(history.size(0), -1, -1)
        return torch.cat((clean, products, valid.to(history.dtype), position), dim=-1)

    def forward(self, power, history):
        encoded = self.encoder(self.features(power, history))
        if self.variant == 'c':
            # Partially observed hours remain usable; entirely missing hours do not.
            valid = (torch.isfinite(power) | torch.isfinite(history).any(dim=-1, keepdim=True))
            encoded = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return encoded
