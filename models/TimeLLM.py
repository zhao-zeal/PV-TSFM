"""Time-LLM official reprogramming + prompt + frozen LLM architecture."""

from layers.timellm.model import Model as OfficialModel


class Model(OfficialModel):
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
