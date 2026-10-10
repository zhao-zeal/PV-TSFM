"""Frozen Chronos-2 adapter for MMSP power and native NWP covariates."""

from dataclasses import dataclass
from pvtsfm.paths import PRETRAINED_DIR

import torch
from chronos import Chronos2Pipeline

from basicts.configs import BasicTSModelConfig


@dataclass
class ChronosC0Config(BasicTSModelConfig):
    model_path: str = str(PRETRAINED_DIR / 'chronos-2')
    input_len: int = 24
    output_len: int = 24
    batch_size: int = 64
    nwp_mode: str = 'none'
    cross_learning: bool = False


class ChronosC0(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pipeline = Chronos2Pipeline.from_pretrained(
            config.model_path, device_map='cpu', local_files_only=True)
        self.backbone = self.pipeline.model
        self.backbone.requires_grad_(False)

    @torch.no_grad()
    def forward(self, inputs, historical_nwp=None, future_nwp=None):
        # Chronos's pinned-memory DataLoader takes CPU contexts and moves them to the backbone's device.
        context = inputs.transpose(1, 2).cpu()
        if self.config.nwp_mode != 'none':
            history = historical_nwp.cpu()
            future = future_nwp.cpu()
            context = []
            for index, target in enumerate(inputs.cpu()):
                past = history[index] if self.config.nwp_mode in ('history', 'history_future') else torch.full_like(history[index], float('nan'))
                task = {'target': target[:, 0], 'past_covariates': {
                    f'nwp_{channel:02d}': past[:, channel] for channel in range(past.shape[1])}}
                if self.config.nwp_mode in ('future', 'history_future'):
                    task['future_covariates'] = {
                        f'nwp_{channel:02d}': future[index, :, channel] for channel in range(future.shape[2])}
                context.append(task)
        quantiles, _ = self.pipeline.predict_quantiles(
            context, prediction_length=self.config.output_len,
            quantile_levels=[0.1, 0.5, 0.9], batch_size=self.config.batch_size,
            cross_learning=self.config.cross_learning,
            context_length=None, limit_prediction_length=False)
        prediction = torch.stack([q[0, :, 1] for q in quantiles]).unsqueeze(-1)
        return {'prediction': prediction.to(inputs.device)}
