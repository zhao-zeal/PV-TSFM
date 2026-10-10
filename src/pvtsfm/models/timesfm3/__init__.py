"""Frozen local TimesFM 3.0 with the official benchmark evaluation settings."""

from dataclasses import dataclass
from pvtsfm.paths import PRETRAINED_DIR

import numpy as np
import torch
from timesfm3 import TimesFM3Evaluator

from basicts.configs import BasicTSModelConfig


@dataclass
class TimesFMC0Config(BasicTSModelConfig):
    model_path: str = str(PRETRAINED_DIR / 'timesfm-3.0-pytorch')
    input_len: int = 24
    output_len: int = 24
    batch_size: int = 64
    use_symmetric_averaging: bool = True
    make_positive: bool = True
    sort_quantiles: bool = True
    use_znorm: bool = False
    padding_mode: str = 'none'
    return_quantiles: bool = True


class TimesFMC0(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.forecaster = TimesFM3Evaluator.from_pretrained(
            config.model_path, device='cpu', local_files_only=True,
            per_core_batch_size=config.batch_size,
        )
        self.backbone = self.forecaster.model
        self.backbone.requires_grad_(False)

    @torch.no_grad()
    def forward(self, inputs):
        self.forecaster.device = inputs.device
        contexts = [row[:, 0] for row in inputs.float().cpu().numpy()]
        forecasts = self.forecaster.predict_batch(
            contexts, horizon=self.config.output_len,
            use_symmetric_averaging=self.config.use_symmetric_averaging,
            make_positive=self.config.make_positive,
            sort_quantiles=self.config.sort_quantiles,
            use_znorm=self.config.use_znorm,
            padding_mode=self.config.padding_mode,
            return_quantiles=self.config.return_quantiles,
        )
        prediction = np.stack([forecast.forecast for forecast in forecasts])
        return {'prediction': torch.as_tensor(prediction, device=inputs.device).unsqueeze(-1)}
