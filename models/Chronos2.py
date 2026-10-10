"""Frozen Chronos-2, using the same MMSP covariate contract as trainable models."""

import torch
from chronos import Chronos2Pipeline


class Model(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.config = configs
        self.pipeline = Chronos2Pipeline.from_pretrained(
            configs.model_path, device_map='cpu', local_files_only=True)
        self.backbone = self.pipeline.model
        self.backbone.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        context = x_enc.transpose(1, 2).cpu()
        mode = self.config.nwp_mode
        if mode != 'none':
            history, future = covariates['historical_nwp'].cpu(), covariates['future_nwp'].cpu()
            context = []
            for index, target in enumerate(x_enc.cpu()):
                past = history[index] if mode in ('history', 'history_future') else torch.full_like(history[index], float('nan'))
                task = {'target': target[:, 0], 'past_covariates': {
                    f'nwp_{channel:02d}': past[:, channel] for channel in range(past.shape[1])}}
                if mode in ('future', 'history_future'):
                    task['future_covariates'] = {
                        f'nwp_{channel:02d}': future[index, :, channel] for channel in range(future.shape[2])}
                context.append(task)
        quantiles, _ = self.pipeline.predict_quantiles(
            context, prediction_length=self.config.pred_len, quantile_levels=[0.1, 0.5, 0.9],
            batch_size=self.config.eval_batch_size, cross_learning=False,
            context_length=None, limit_prediction_length=False)
        return torch.stack([q[0, :, 1] for q in quantiles]).unsqueeze(-1).to(x_enc.device)
