"""A01 with horizon-dependent soft NWP weighting after native normalization.

Zero weight removes the normalized value, not the covariate token or its mask.
This is soft weighting, not an exact differentiable version of channel deletion.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.Chronos2Adapter import Model as A01


class Model(A01):
    def __init__(self, config):
        super().__init__(config)
        self.selector = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 15))
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.constant_(self.selector[-1].bias, math.log(19))  # Initially 0.95 open.
        self.gate_mode = getattr(config, 'gate_mode', 'dynamic')
        fixed = getattr(config, 'fixed_gate_input', None)
        self.register_buffer('fixed_gate_input', torch.tensor(fixed if fixed is not None else [0.] * 64), persistent=False)

    @staticmethod
    def gate_inputs(power, history, future):
        summary = torch.cat([power.mean(1, keepdim=True), power.std(1, keepdim=True, unbiased=False),
                             power[:, -1:], history.mean(1), history.std(1, unbiased=False),
                             history[:, -1]], dim=-1)
        horizon = torch.arange(1, 25, device=power.device, dtype=power.dtype).view(1, 24, 1) / 24
        inputs = torch.cat([summary[:, None].expand(-1, 24, -1), future,
                            horizon.expand(len(power), -1, -1)], dim=-1)
        return inputs

    def gates(self, power, history, future):
        if self.gate_mode == 'fixed':
            # Same MLP and parameter count; a training-only constant removes all
            # sample/horizon conditioning. Weights remain trainable in stage 1.
            return self.selector(self.fixed_gate_input).sigmoid().view(1, 1, 15).expand(len(power), 24, 15)
        return self.selector(self.gate_inputs(power, history, future)).sigmoid()

    @staticmethod
    def weight_values(patched, weights):
        time, values, mask = patched.chunk(3, dim=-1)
        return torch.cat([time, values * weights, mask], dim=-1)

    def encode_inputs(self, power, history, future, gates=None):
        if getattr(self.config, 'nwp_indices', None) not in (None, list(range(15))):
            raise ValueError('Dynamic gate requires the original 15 NWP channels')
        weights = self.gates(power, history, future) if gates is None else gates
        batch = len(power)
        group = 16
        context = torch.cat([power[:, None], history.transpose(1, 2)], dim=1).reshape(batch * group, 24)
        future_values = torch.cat([torch.full_like(future[:, :, :1].transpose(1, 2), torch.nan),
                                   future.transpose(1, 2)], dim=1).reshape(batch * group, 24)
        backbone = self.backbone
        patched, mask, loc_scale = backbone._prepare_patched_context(context)
        history_weights = torch.cat([torch.ones_like(weights[:, 0, :1]), weights.mean(1)], dim=1)
        patched = self.weight_values(patched, history_weights.reshape(batch * group, 1, 1))
        embeds = backbone.input_patch_embedding(patched)
        if backbone.chronos_config.use_reg_token:
            ids = torch.full((batch * group, 1), backbone.config.reg_token_id,
                             dtype=torch.long, device=power.device)
            embeds = torch.cat([embeds, backbone.shared(ids)], dim=-2)
            mask = torch.cat([mask.to(backbone.dtype), torch.ones_like(ids).to(backbone.dtype)], dim=-1)
        patched_future, _ = backbone._prepare_patched_future(
            future_values, None, loc_scale, 2, batch * group)
        future_weights = torch.cat([torch.ones_like(weights[:, :, :1]), weights], dim=-1)
        size = backbone.chronos_config.output_patch_size
        future_weights = F.pad(future_weights.transpose(1, 2), (0, 2 * size - 24), value=1)
        patched_future = self.weight_values(patched_future, future_weights.reshape(batch * group, 2, size))
        embeds = torch.cat([embeds, backbone.input_patch_embedding(patched_future)], dim=-2)
        mask = torch.cat([mask, torch.ones(batch * group, 2, device=power.device, dtype=backbone.dtype)], dim=-1)
        encoded = backbone.encoder(inputs_embeds=embeds, attention_mask=mask,
            group_ids=torch.arange(batch, device=power.device).repeat_interleave(group), output_attentions=False)
        targets = torch.arange(batch, device=power.device) * group
        return encoded[0][targets, -2:], loc_scale[0][targets], loc_scale[1][targets]
