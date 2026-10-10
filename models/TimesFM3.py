"""Retain the accepted optional TimesFM 3.0 official evaluation settings."""

import numpy as np
import torch
from timesfm3 import TimesFM3Evaluator


class Model(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.config = configs
        self.forecaster = TimesFM3Evaluator.from_pretrained(
            configs.model_path, device='cpu', local_files_only=True,
            per_core_batch_size=configs.eval_batch_size)
        self.backbone = self.forecaster.model
        self.backbone.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        self.forecaster.device = x_enc.device
        forecasts = self.forecaster.predict_batch(
            [row[:, 0] for row in x_enc.cpu().numpy()], horizon=self.config.pred_len,
            use_symmetric_averaging=True, make_positive=True, sort_quantiles=True,
            use_znorm=False, padding_mode='none', return_quantiles=True)
        return torch.as_tensor(np.stack([f.forecast for f in forecasts]),
                               device=x_enc.device).unsqueeze(-1)
