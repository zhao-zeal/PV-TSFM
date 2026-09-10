import copy

import torch
from torch import nn

from pv_tsfm.models.identity import parameter_hash
from pv_tsfm.models.lora import apply_shared_pv_lora


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4, bias=False)
        self.k = nn.Linear(4, 4, bias=False)
        self.v = nn.Linear(4, 4, bias=False)
        self.o = nn.Linear(4, 4, bias=False)

    def forward(self, value):
        return self.o(self.q(value) + self.k(value) + self.v(value))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = Attention()

    def forward(self, value):
        return self.self_attention(value)


class OutputPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layer = nn.Linear(4, 1, bias=False)

    def forward(self, value):
        return self.output_layer(value)


class TinyChronosShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.output_patch_embedding = OutputPatch()

    def forward(self, value):
        return self.output_patch_embedding(self.block(value))


def test_only_lora_parameters_change_and_original_hash_is_stable():
    torch.manual_seed(1)
    original = TinyChronosShape()
    original_hash = parameter_hash(original, include_lora=False)
    adapted, report = apply_shared_pv_lora(copy.deepcopy(original))
    assert report["matched_module_count"] == 5
    assert report["trainable_parameter_count"] > 0
    assert all("lora_" in name for name in report["trainable_parameter_names"])
    before_lora = parameter_hash(adapted, include_lora=True)

    optimizer = torch.optim.AdamW((p for p in adapted.parameters() if p.requires_grad), lr=1e-3)
    loss = adapted(torch.randn(3, 4)).square().mean()
    loss.backward()
    optimizer.step()

    assert parameter_hash(adapted, include_lora=False) == original_hash
    assert parameter_hash(adapted, include_lora=True) != before_lora
    assert all(parameter.grad is None for name, parameter in adapted.named_parameters() if "lora_" not in name)
