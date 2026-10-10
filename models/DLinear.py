"""TSLib model interface; preserve the accepted power + coordinate baseline."""

import torch
from torch import nn

from layers.dlinear import Model as Backbone


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.backbone = Backbone(seq_len=configs.seq_len, pred_len=configs.pred_len,
                                 enc_in=3, moving_avg=configs.moving_avg,
                                 individual=configs.individual)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        coords = covariates['site_coords'].unsqueeze(1).expand(-1, x_enc.shape[1], -1)
        return self.backbone(torch.cat([x_enc, coords], dim=-1), None, None, None)[..., :1]
