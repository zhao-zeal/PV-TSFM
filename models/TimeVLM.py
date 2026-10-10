"""Official Time-VLM CLIP full-shot architecture on the common MMSP interface."""

from layers.timevlm.model import Model as OfficialModel


class Model(OfficialModel):
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
