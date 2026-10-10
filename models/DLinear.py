"""Official DLinear; archived FusionSF variant remains explicitly selectable."""

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.legacy = configs.baseline_config == 'legacy_fusionsf'
        if self.legacy:
            from layers.dlinear import Model as Backbone
            self.backbone = Backbone(seq_len=configs.seq_len, pred_len=configs.pred_len,
                                     enc_in=3, moving_avg=configs.moving_avg,
                                     individual=configs.individual)
        else:
            from layers.dlinear_official.model import Model as Backbone
            self.backbone = Backbone(configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        if self.legacy:
            coords = covariates['site_coords'].unsqueeze(1).expand(-1, x_enc.shape[1], -1)
            return self.backbone(torch.cat([x_enc, coords], dim=-1), None, None, None)[..., :1]
        return self.backbone(x_enc)
