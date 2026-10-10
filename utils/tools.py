"""Checkpoint loading and validation-based stopping shared by experiments."""

import torch


def load_model_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best = float('inf')
        self.bad_epochs = 0

    def update(self, value):
        improved = value < self.best
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
