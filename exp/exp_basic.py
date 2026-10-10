"""TSLib's experiment base: model construction and device selection."""

import importlib

import torch


MODEL_MODULES = {name: f'models.{name}' for name in (
    'DLinear', 'PatchTST', 'FusionSF', 'CrossUnet', 'TimeXer', 'Chronos2', 'TimesFM3',
    'ChronosX', 'TimeVLM')}


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if args.use_gpu else 'cpu')
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        return importlib.import_module(MODEL_MODULES[self.args.model]).Model(self.args).float()
