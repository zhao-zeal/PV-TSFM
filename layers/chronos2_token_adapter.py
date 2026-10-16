"""Controlled T1/T2 MLP and T3 Cross-Unet-inspired residuals on native tokens."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def position_encoding(position, width):
    frequency = torch.exp(torch.arange(0, width, 2, device=position.device,
                                      dtype=position.dtype) * (-math.log(10000.) / width))
    angles = position[..., None] * frequency
    return torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)


def history_correlation(power, history):
    # Only the common latest 24 hours; no Cross-Unet extra PV history block.
    values = torch.cat((power, history), -1).double()
    values = values - values.mean(1, keepdim=True)
    values = values / values.square().sum(1, keepdim=True).sqrt().clamp_min(1e-12)
    return (values.transpose(1, 2) @ values).clamp(-1, 1).float()


class CrossBlock(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.temporal = nn.MultiheadAttention(width, 4, dropout=.05, batch_first=True)
        self.time_norm = nn.LayerNorm(width)
        self.channel = nn.Linear(width, width)
        self.channel_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(.05), nn.Linear(128, width))
        self.ff_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(.05)

    def forward(self, x, correlation):
        batch, channels, patches, width = x.shape
        temporal = x.reshape(batch * channels, patches, width)
        attended = self.temporal(temporal, temporal, temporal, need_weights=False)[0]
        x = self.time_norm(temporal + self.dropout(attended)).reshape_as(x)
        mixed = torch.einsum('bcj,bjpd->bcpd', correlation, x)
        x = self.channel_norm(x + self.dropout(self.channel(mixed)))
        return self.ff_norm(x + self.dropout(self.ff(x)))


class CrossMixer(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.pcorr = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 256))
        self.fine = CrossBlock(width)
        self.merge = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width))
        self.coarse = CrossBlock(width)

    def forward(self, x, correlation):
        weights = self.pcorr(correlation.flatten(1)).reshape(-1, 16, 16).softmax(-1)
        fine = self.fine(x, weights)
        # Native history/future each have an even patch count in the approved contexts.
        coarse = self.merge(fine.reshape(*fine.shape[:2], fine.size(2) // 2, -1))
        coarse = self.coarse(coarse, weights)
        return torch.cat((fine.flatten(1, 2), coarse.flatten(1, 2)), 1)


class TokenResidual(nn.Module):
    def __init__(self, variant, mlp_width):
        super().__init__()
        self.variant = variant
        # Common components initialized identically before the variant-specific mixer.
        self.project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64))
        self.metadata = nn.Linear(3, 64, bias=False)
        self.weather = nn.Linear(30, 64)
        self.readout = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.mixer = (CrossMixer() if variant == 'T3' else
                      nn.Sequential(nn.LayerNorm(64), nn.Linear(64, mlp_width), nn.GELU(),
                                    nn.Dropout(.05), nn.Linear(mlp_width, 64)))

    def forward(self, tokens, future, correlation, metadata, positions):
        # metadata: history/future type, real-time coverage, observed-value fraction.
        if self.variant == 'T1':
            tokens, metadata, positions = tokens[:, :1, -2:], metadata[:, :1, -2:], positions[-2:]
        x = self.project(tokens) + self.metadata(metadata)
        x = x + position_encoding(positions, 64)[None, None]
        channels = torch.arange(x.size(1), device=x.device, dtype=x.dtype)
        x = x + position_encoding(channels, 64)[None, :, None]
        if self.variant == 'T3':
            memory = self.mixer(x, correlation)
        else:
            memory = (x + self.mixer(x)).flatten(1, 2)
        missing = future.isnan()
        weather = self.weather(torch.cat((future.masked_fill(missing, 0), missing.to(future.dtype)), -1))
        hours = torch.arange(future.size(1), device=x.device, dtype=x.dtype)
        query = weather + position_encoding(hours, 64)[None]
        attention = (query @ memory.transpose(-1, -2) / 8.).softmax(-1)
        return self.readout(torch.cat((attention @ memory, weather), -1))


def matched_mlp_width():
    # Parameter arithmetic only, never selected using validation or test performance.
    with torch.random.fork_rng(devices=[]):
        count = sum(p.numel() for p in CrossMixer().parameters())
    # LayerNorm(64) + Linear(64,w) + Linear(w,64) = 192 + 129*w.
    return round((count - 192) / 129)


def patch_metadata(power, history, future, patch_size=16):
    batch, length, _ = power.shape
    channels = history.size(-1) + 1
    context = torch.cat((power, history), -1).transpose(1, 2)
    unknown = torch.full_like(future[..., :1], torch.nan)
    forecast = torch.cat((unknown, future), -1).transpose(1, 2)
    left, right = (-length) % patch_size, (-future.size(1)) % patch_size
    observed = torch.cat((F.pad(context.isfinite().float(), (left, 0)),
                          F.pad(forecast.isfinite().float(), (0, right))), -1)
    coverage = torch.cat((F.pad(torch.ones_like(context), (left, 0)),
                          F.pad(torch.ones_like(forecast), (0, right))), -1)
    observed = observed.reshape(batch, channels, -1, patch_size).mean(-1)
    coverage = coverage.reshape(batch, channels, -1, patch_size).mean(-1)
    n_history = (length + left) // patch_size
    kind = torch.zeros_like(observed)
    kind[..., n_history:] = 1
    positions = torch.arange(-n_history, observed.size(-1) - n_history,
                             device=power.device, dtype=power.dtype) * patch_size
    return torch.stack((kind, coverage, observed), -1), positions
