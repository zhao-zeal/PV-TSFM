import copy

import torch
from torch import nn

from pv_tsfm.models.identity import parameter_hash


def test_each_seed_or_fold_starts_from_same_original_checkpoint():
    torch.manual_seed(11)
    checkpoint = nn.Linear(8, 2)

    def reset():
        return copy.deepcopy(checkpoint)

    fold_a = reset()
    fold_b = reset()
    initial = parameter_hash(checkpoint)
    assert parameter_hash(fold_a) == initial == parameter_hash(fold_b)
    with torch.no_grad():
        fold_a.weight.add_(1)
    assert parameter_hash(fold_a) != initial
    assert parameter_hash(fold_b) == initial
