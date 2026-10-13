"""Frozen native Chronos-2 + B131 + E11/E12 output residual, with shared MMSP input."""

import torch
from chronos import Chronos2Pipeline
from torch import nn

from layers.chronos2_adapter import ForecastOnlyBottleneckAdapter, residual_readout


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.config = configs
        pipeline = Chronos2Pipeline.from_pretrained(
            configs.model_path, device_map='cpu', local_files_only=True)
        self.backbone = pipeline.model.eval().requires_grad_(False)
        self.adapter = ForecastOnlyBottleneckAdapter()
        self.readout = residual_readout(configs.adapter_version)
        self.stage = 'base'
        self.median_index = int((self.backbone.quantiles.float() - 0.5).abs().argmin())

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self.stage != 'base':
            self.adapter.eval()
        return self

    @torch.no_grad()
    def encode_inputs(self, power, history, future):
        batch_size, context_length = power.shape
        group_size = 1 + history.shape[-1]
        context = torch.cat([power.unsqueeze(1), history.transpose(1, 2)], dim=1)
        unknown_target = torch.full_like(future[:, :, :1].transpose(1, 2), torch.nan)
        future = torch.cat([unknown_target, future.transpose(1, 2)], dim=1)
        indices = torch.arange(batch_size, device=power.device) * group_size
        patch_count = 2
        encoded, (loc, scale), _, _ = self.backbone.encode(
            context=context.reshape(batch_size * group_size, context_length),
            future_covariates=future.reshape(batch_size * group_size, self.config.pred_len),
            group_ids=torch.arange(batch_size, device=power.device).repeat_interleave(group_size),
            num_output_patches=patch_count,
        )
        return encoded[0][indices, -patch_count:], loc[indices], scale[indices]

    def median_forecast(self, embeds, loc, scale):
        batch_size, patches, _ = embeds.shape
        patch_size = self.backbone.chronos_config.output_patch_size
        quantiles = self.backbone.num_quantiles
        values = self.backbone.output_patch_embedding(embeds)
        values = values.reshape(batch_size, patches, quantiles, patch_size)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, -1)
        values = self.backbone.instance_norm.inverse(values, (loc, scale))
        return values.reshape(batch_size, quantiles, -1)[:, self.median_index, :24]

    def forward_features(self, embeds, loc, scale):
        adapted = self.adapter(embeds)
        forecast = self.median_forecast(adapted, loc, scale)
        if self.stage != 'base':
            forecast = forecast + self.readout(adapted.flatten(1))
        return forecast.unsqueeze(-1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        embeds, loc, scale = self.encode_inputs(
            x_enc[..., 0], covariates['historical_nwp'], covariates['future_nwp'])
        return self.forward_features(embeds, loc, scale)
