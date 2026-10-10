"""Official ChronosX T5 + IIB/OIB; quantized CE training, sampled forecasts."""

import torch
from torch import nn
from torch.nn import functional as F
from chronos import ChronosConfig
from transformers import AutoConfig, GenerationConfig

from layers.chronosx.model import ChronosX


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.args = configs
        config = AutoConfig.from_pretrained(configs.model_path, local_files_only=True)
        tokenizer_config = dict(config.chronos_config)
        tokenizer_config['prediction_length'] = configs.pred_len
        self.tokenizer = ChronosConfig(**tokenizer_config).create_tokenizer()
        self.backbone = ChronosX.set_state(
            num_covariates=30, covariate_injection=configs.covariate_injection,
            hidden_dim=configs.injection_hidden_dim, num_layers=configs.injection_layers,
            vocab_size=config.vocab_size, model_dim=config.d_model,
        ).from_pretrained(configs.model_path, local_files_only=True)
        self.backbone.initialize_blocks()
        self.backbone.config.pad_token_id = self.backbone.generation_config.pad_token_id = 0
        self.backbone.config.eos_token_id = self.backbone.generation_config.eos_token_id = 1
        self.backbone.freeze('all')
        self.backbone.unfreeze('injection_block')

    def _covariates(self, covariates):
        # Official loader concatenates values and missing-value indicators.
        history, future = (covariates[key] for key in ('historical_nwp', 'future_nwp'))
        history = torch.cat([torch.nan_to_num(history, nan=-1.), history.isnan().float()], -1)
        future = torch.cat([torch.nan_to_num(future, nan=-1.), future.isnan().float()], -1)
        scale = history.abs().mean(1, keepdim=True).clamp_min(1.)
        history, future = history / scale, future / scale
        # Extra row corresponds to the Chronos EOS token, as in the author loader.
        history = F.pad(history, (0, 0, 0, 1))
        future = F.pad(future, (0, 0, 0, 1))
        return history, future

    def token_loss(self, x_enc, targets, covariates, augment=False):
        history, future = self._covariates(covariates)
        context, target = x_enc[..., 0].clone(), targets[..., 0].clone()
        if augment:
            probabilities = torch.rand(context.size(0), 1, device=context.device) * self.args.chronosx_drop_prob
            context.masked_fill_(torch.rand_like(context) < probabilities, float('nan'))
            target.masked_fill_(torch.rand_like(target) < probabilities, float('nan'))
        # Official tokenizer stores bucket boundaries on CPU.
        token_ids, mask, scale = self.tokenizer.context_input_transform(context.cpu())
        labels, label_mask = self.tokenizer.label_input_transform(target.cpu(), scale)
        labels = labels.masked_fill(~label_mask.bool(), -100).to(x_enc.device)
        output = self.backbone(input_ids=token_ids.to(x_enc.device), attention_mask=mask.to(x_enc.device),
                               past_covariates=history, future_covariates=future,
                               labels=labels)
        # Official OIB returns loss only in train mode; validation uses the same CE.
        return F.cross_entropy(output.logits.flatten(0, 1), labels.flatten(), ignore_index=-100)

    @torch.no_grad()
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, covariates=None):
        training = self.backbone.training
        self.backbone.eval()  # Official sampling path does not compute a label loss.
        history, future = self._covariates(covariates)
        tokens, mask, scale = self.tokenizer.context_input_transform(x_enc[..., 0].cpu())
        samples = self.backbone.generate(
            input_ids=tokens.to(x_enc.device), attention_mask=mask.to(x_enc.device), past_covariates=history,
            future_covariates=future,
            generation_config=GenerationConfig(
                min_new_tokens=self.args.pred_len, max_new_tokens=self.args.pred_len,
                do_sample=True, num_return_sequences=self.args.num_samples,
                eos_token_id=1, pad_token_id=0),
        )[..., 1:].reshape(x_enc.size(0), self.args.num_samples, self.args.pred_len)
        values = self.tokenizer.output_transform(samples.cpu(), scale)
        self.backbone.train(training)
        return values.quantile(0.5, dim=1).unsqueeze(-1).to(x_enc.device)
