"""Frozen accepted E11 model plus a trainable OIB-inspired Weather Adapter."""

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from layers.chronos2_weather_adapter import HorizonWeatherResidualAdapter
from models.Chronos2Adapter import Model as LinearAdapterModel


ROOT = Path(__file__).resolve().parents[1]


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        base_config = copy.copy(configs)
        base_config.adapter_version = 'b131_linear_v1'
        self.base = LinearAdapterModel(base_config)
        checkpoint_path = Path(configs.adapter_base_checkpoint).resolve()
        accepted = json.loads((ROOT / 'reports/Chronos2Adapter_two_protocols_seed42_20261011/runs.json').read_text())
        entries = accepted['acceptance']['runs']
        reference = next((row for row in entries if row['protocol'] == configs.protocol
                          and row['adapter_version'] == 'b131_linear_v1'), None)
        if reference is None or reference['status'] != 'passed_full_training_test_artifact_acceptance':
            raise ValueError('The same-protocol Linear Adapter must have passed artifact acceptance')
        if checkpoint_path != Path(reference['checkpoint_path']).resolve():
            raise ValueError('Base checkpoint must be the accepted same-protocol Linear Adapter')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        saved = checkpoint['config']
        for key in ('protocol', 'seed', 'seq_len', 'pred_len', 'nwp_mode', 'modalities'):
            if saved[key] != getattr(configs, key):
                raise ValueError(f'Base checkpoint configuration differs: {key}')
        if checkpoint['stage'] != 'b131_linear_v1':
            raise ValueError('Expected the accepted Linear readout checkpoint')
        if Path(saved['model_path']).resolve() != Path(configs.model_path).resolve():
            raise ValueError('Pretrained model path differs from the accepted base')
        self.base.adapter.load_state_dict(checkpoint['adapter_state_dict'], strict=True)
        self.base.readout.load_state_dict(checkpoint['readout_state_dict'], strict=True)
        self.base.stage = 'b131_linear_v1'
        self.base.eval().requires_grad_(False)
        self.weather_adapter = HorizonWeatherResidualAdapter(
            patch_size=self.base.backbone.chronos_config.output_patch_size)
        self.base_provenance = {
            'checkpoint_path': str(checkpoint_path),
            'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'base_version': 'b131_linear_v1', 'base_best_epoch': checkpoint['epoch'],
            'base_experiment_label': 'A01', 'historical_name': 'E11',
            'acceptance_path': str(ROOT / 'reports/Chronos2Adapter_two_protocols_seed42_20261011/runs.json'),
            'official_inspiration': 'ChronosX OIB; power residual rather than token logits',
        }

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward_features(self, embeds, loc, scale, future_nwp):
        with torch.no_grad():
            adapted = self.base.adapter(embeds)
            forecast = self.base.forward_features(embeds, loc, scale)
        return forecast + self.weather_adapter(adapted, future_nwp)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        embeds, loc, scale = self.base.encode_inputs(
            x_enc[..., 0], covariates['historical_nwp'], covariates['future_nwp'])
        return self.forward_features(embeds, loc, scale, covariates['future_nwp'])
