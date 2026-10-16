"""A05 decoder built from the active Cross-Unet P-Corr/decoder computations."""

import math

import torch
from torch import nn


def position_encoding(position, width):
    frequency = torch.exp(torch.arange(0, width, 2, device=position.device,
                                      dtype=position.dtype) * (-math.log(10000.) / width))
    angles = position[..., None] * frequency
    return torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)


def official_pcorr_features(power, historical_nwp, correlation_history):
    """Cross-Unet's nonnegative Pearson vector before its two learned projections."""
    if power.shape[1:] != (24, 1) or historical_nwp.shape[1:] != (24, 15):
        raise ValueError('A05 P-Corr requires 24-hour power and 15 historical NWP variables')
    if correlation_history.shape[1:] != (24, 1):
        raise ValueError('A05 P-Corr requires the previous 24-hour power block')
    values = torch.cat((historical_nwp, correlation_history, power), -1).double()
    centered = values - values.mean(1, keepdim=True)
    normalized = centered / centered.square().sum(1, keepdim=True).sqrt().clamp_min(1e-12)
    correlation = normalized.transpose(1, 2) @ normalized
    # Official ordering is [NWP15, previous-power, current-power], then first 16 vs last.
    return torch.nan_to_num(correlation[:, :-1, -1], nan=0., posinf=0., neginf=0.).clamp_min(0.).float()


class OfficialPCorr(nn.Module):
    """Exact nonlinear 16 -> 16x16 P-Corr projection used by Cross-Unet."""

    def __init__(self, channels=16):
        super().__init__()
        self.channel_projection = nn.Sequential(
            nn.Linear(channels, channels * 4, bias=False), nn.Sigmoid(),
            nn.Linear(channels * 4, channels, bias=False))
        self.matrix_projection = nn.Sequential(
            nn.Linear(channels * channels, channels * channels * 4, bias=False), nn.Sigmoid(),
            nn.Linear(channels * channels * 4, channels * channels, bias=False))

    def forward(self, features):
        if features.ndim != 2 or features.size(1) != 16:
            raise ValueError('P-Corr features must have shape [B,16]')
        projected = self.channel_projection(features).unsqueeze(-1).expand(-1, -1, 16)
        projected = self.matrix_projection(projected.reshape(features.size(0), -1))
        return projected.softmax(-1).reshape(-1, 16, 16)


class ActiveTwoStageBlock(nn.Module):
    """The time-attention, P-Corr mixing and FF path actually used upstream."""

    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.time_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Linear(feedforward, width))
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, correlation=None):
        batch, channels, segments, width = x.shape
        flattened = x.reshape(batch * channels, segments, width)
        temporal = self.time_attention(flattened, flattened, flattened, need_weights=False)[0]
        temporal = temporal.reshape_as(x)
        mixed = 0.
        if correlation is not None:
            mixed = torch.bmm(correlation, x.reshape(batch, channels, -1)).reshape_as(x)
        x = self.norm1(x + self.dropout(temporal + mixed))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class TemporalRefine(nn.Module):
    """Per-channel temporal refinement used to construct a real feature pyramid."""

    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.block = ActiveTwoStageBlock(width, heads, feedforward, dropout)

    def forward(self, x):
        return self.block(x, None)


class SegmentMerge(nn.Module):
    """Cross-Unet SegMerging with an explicit factor of two."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(2 * width)
        self.projection = nn.Linear(2 * width, width)

    def forward(self, x):
        if x.size(2) % 2:
            x = torch.cat((x, x[:, :, -1:]), 2)
        merged = torch.cat((x[:, :, 0::2], x[:, :, 1::2]), -1)
        return self.projection(self.norm(merged))


class MCDecoderLayer(nn.Module):
    """One active Cross-Unet decoder layer and its segment prediction head."""

    def __init__(self, width, heads, feedforward, segment_length, dropout):
        super().__init__()
        self.self_attention = ActiveTwoStageBlock(width, heads, feedforward, dropout)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.output_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.segment_head = nn.Linear(width, segment_length)
        nn.init.zeros_(self.segment_head.weight)
        nn.init.zeros_(self.segment_head.bias)

    def forward(self, query, memory, correlation):
        query = self.self_attention(query, correlation)
        batch, channels, segments, width = query.shape
        q = query.reshape(batch * channels, segments, width)
        m = memory.reshape(batch * channels, memory.size(2), width)
        attended = self.cross_attention(q, m, m, need_weights=False)[0]
        x = self.cross_norm(q + self.dropout(attended))
        x = self.output_norm(x + self.feed_forward(x)).reshape_as(query)
        return x, self.segment_head(x)


class A05Head(nn.Module):
    """A05-a/b/c residual head over native non-REG Chronos-2 states."""

    def __init__(self, variant, width=128, heads=4, feedforward=256, dropout=.05):
        super().__init__()
        if variant not in ('a', 'b', 'c'):
            raise ValueError('A05 variant must be a, b or c')
        self.variant = variant
        self.width = width
        self.project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.metadata = nn.Linear(3, width, bias=False)
        self.pcorr = OfficialPCorr() if variant in ('b', 'c') else None
        if variant == 'c':
            self.refine = TemporalRefine(width, heads, feedforward, dropout)
            self.merge1 = SegmentMerge(width)
            self.merge2 = SegmentMerge(width)
            self.bottleneck = TemporalRefine(width, heads, feedforward, dropout)
        self.decoder = nn.ModuleList([
            MCDecoderLayer(width, heads, feedforward, segment_length=16, dropout=dropout)
            for _ in range(4)])

    def forward(self, tokens, metadata, positions, pcorr_features):
        if tokens.shape[1:] != (16, 4, 768) or metadata.shape[1:] != (16, 4, 3):
            raise ValueError('A05 expects tokens [B,16,4,768] and metadata [B,16,4,3]')
        if tuple(positions.shape) != (4,):
            raise ValueError('A05 expects four native patch positions')
        x = self.project(tokens) + self.metadata(metadata)
        x = x + position_encoding(positions.to(x), self.width)[None, None]
        channels = torch.arange(16, device=x.device, dtype=x.dtype)
        x = x + position_encoding(channels, self.width)[None, :, None]
        # Chronos groups PV first; Cross-Unet/P-Corr order is NWP15 then PV.
        order = torch.tensor([*range(1, 16), 0], device=x.device)
        x = x.index_select(1, order)
        query = x[:, :, 2:]
        if self.variant == 'c':
            fine = self.refine(x)
            half = self.merge1(fine)
            coarse = self.merge2(half)
            bottleneck = self.bottleneck(coarse)
            memories = (bottleneck, half, fine, x)
        else:
            memories = (x,) * 4
        correlation = self.pcorr(pcorr_features) if self.pcorr is not None else None
        predictions = []
        for layer, memory in zip(self.decoder, memories):
            query, prediction = layer(query, memory, correlation)
            predictions.append(prediction)
        # Four decoder layers vote additively, as in the official implementation.
        residual = torch.stack(predictions).sum(0)  # [B,16,2,16]
        residual = residual.permute(0, 2, 3, 1).reshape(tokens.size(0), 32, 16)
        return residual[:, :24, -1:]
