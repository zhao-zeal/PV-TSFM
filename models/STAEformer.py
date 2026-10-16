"""Single-site MMSP wrapper for the published STAEformer architecture."""

import torch
from torch import nn

from layers.staeformer.official_model import STAEformer


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        model_dim = (configs.stae_input_embedding_dim + configs.stae_tod_embedding_dim
                     + configs.stae_dow_embedding_dim + configs.stae_spatial_embedding_dim
                     + configs.stae_adaptive_embedding_dim)
        if model_dim != configs.d_model:
            raise ValueError('STAEformer embedding dimensions must sum to d_model')
        self.backbone = STAEformer(
            num_nodes=1,
            in_steps=configs.seq_len,
            out_steps=configs.pred_len,
            steps_per_day=24,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=configs.stae_input_embedding_dim,
            tod_embedding_dim=configs.stae_tod_embedding_dim,
            dow_embedding_dim=configs.stae_dow_embedding_dim,
            spatial_embedding_dim=configs.stae_spatial_embedding_dim,
            adaptive_embedding_dim=configs.stae_adaptive_embedding_dim,
            feed_forward_dim=configs.d_ff,
            num_heads=configs.n_heads,
            num_layers=configs.e_layers,
            dropout=configs.dropout,
            use_mixed_proj=True,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        hour = torch.round((x_mark_enc[..., 0] + 0.5) * 23).clamp(0, 23) / 24
        weekday = torch.round((x_mark_enc[..., 1] + 0.5) * 6).clamp(0, 6)
        inputs = torch.stack((x_enc[..., 0], hour, weekday), dim=-1).unsqueeze(2)
        return self.backbone(inputs).squeeze(2)
