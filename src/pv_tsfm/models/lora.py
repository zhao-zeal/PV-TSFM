"""Strict E4 LoRA construction; never falls back to full fine-tuning."""

from __future__ import annotations

from importlib.util import find_spec


LORA_TARGET_SUFFIXES = (
    "self_attention.q",
    "self_attention.k",
    "self_attention.v",
    "self_attention.o",
    "output_patch_embedding.output_layer",
)


def enumerate_lora_targets(model) -> list[str]:
    return [
        name for name, _ in model.named_modules()
        if any(name.endswith(suffix) for suffix in LORA_TARGET_SUFFIXES)
    ]


def apply_shared_pv_lora(model):
    if find_spec("peft") is None:
        raise RuntimeError("PEFT is unavailable; E4 must fail and must not fall back to full fine-tuning")
    matched = enumerate_lora_targets(model)
    if not matched:
        raise RuntimeError("LoRA target module match count is zero; E4 identity is invalid")

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(LORA_TARGET_SUFFIXES),
    )
    adapted = get_peft_model(model, config)
    report = lora_identity_report(adapted, matched_modules=matched)
    if not report["trainable_parameter_names"]:
        raise RuntimeError("LoRA produced zero trainable parameters")
    invalid = [name for name in report["trainable_parameter_names"] if "lora_" not in name]
    if invalid:
        raise RuntimeError(f"non-LoRA parameters are trainable: {invalid}")
    return adapted, report


def lora_identity_report(model, *, matched_modules: list[str] | None = None) -> dict:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    total_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    return {
        "configuration": {"r": 8, "alpha": 16, "dropout": 0.0, "bias": "none"},
        "target_module_suffixes": list(LORA_TARGET_SUFFIXES),
        "matched_module_names": matched_modules if matched_modules is not None else enumerate_lora_targets(model),
        "matched_module_count": len(matched_modules) if matched_modules is not None else len(enumerate_lora_targets(model)),
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "trainable_percent": 100.0 * trainable_count / total_count,
    }
