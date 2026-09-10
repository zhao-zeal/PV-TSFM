"""Parameter identity checks used by reset and freezing acceptance tests."""

from __future__ import annotations

import hashlib


def tensor_hash(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(model, *, include_lora: bool = True) -> str:
    parameters = model.named_parameters()
    if not include_lora:
        parameters = ((canonical_base_name(name), value) for name, value in parameters if "lora_" not in name)
    return tensor_hash(parameters)


def canonical_base_name(name: str) -> str:
    while name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    return name.replace(".base_layer.", ".")
