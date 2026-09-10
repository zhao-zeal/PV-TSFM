"""Small explicit trainer around the official Chronos-2 model and loss."""

from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Callable, Iterable

import numpy as np

from pv_tsfm.models import Chronos2Forecaster, apply_shared_pv_lora
from pv_tsfm.models.identity import parameter_hash
from .config import TrainingConfig


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_fresh_trainable_model(config: TrainingConfig, *, device: str = "cuda"):
    import torch

    seed_everything(config.seed)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    pipeline = Chronos2Forecaster.from_pretrained(
        device_map=device, torch_dtype=dtype, local_files_only=True
    ).pipeline
    base_hash = parameter_hash(pipeline.model, include_lora=False)
    if config.method == "E4_shared_lora_cpt":
        model, identity = apply_shared_pv_lora(pipeline.model)
        if parameter_hash(model, include_lora=False) != base_hash:
            raise RuntimeError("attaching LoRA changed original Chronos weights")
    else:
        model = pipeline.model
        model.requires_grad_(True)
        identity = {
            "trainable_parameter_names": [name for name, _ in model.named_parameters()],
            "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
            "total_parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_percent": 100.0,
        }
    return model, identity, base_hash


def train_optimizer_steps(
    model,
    batches: Iterable[dict],
    config: TrainingConfig,
    *,
    output_dir: str | Path,
    validation_callback: Callable[[object, int], dict] | None = None,
) -> list[dict]:
    """Run exact optimizer steps; each yielded batch is one microbatch."""
    import torch

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("training method has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    warmup_steps = int(config.warmup_fraction * config.schedule_steps)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return max(0.0, float(config.schedule_steps - step) / float(max(1, config.schedule_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    iterator = iter(batches)
    history = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config.training_steps + 1):
        losses = []
        for _ in range(config.gradient_accumulation):
            batch = next(iterator)
            context = batch["context"]
            future = batch["future_target"]
            mask = batch.get("context_mask", torch.isfinite(context))
            future_mask = batch.get("future_target_mask", torch.isfinite(future))
            group_ids = torch.arange(context.shape[0], device=context.device)
            result = model(
                context=context,
                context_mask=mask,
                group_ids=group_ids,
                num_output_patches=math.ceil(config.horizon / model.chronos_config.output_patch_size),
                future_target=future,
                future_target_mask=future_mask,
            )
            if result.loss is None or not torch.isfinite(result.loss):
                raise RuntimeError("official Chronos-2 loss is missing or non-finite")
            (result.loss / config.gradient_accumulation).backward()
            losses.append(float(result.loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        row = {"step": step, "loss": float(np.mean(losses)), "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm)}
        if validation_callback is not None and step % 250 == 0:
            row.update(validation_callback(model, step))
            model.train()
        history.append(row)
        if step in {500, 1000}:
            torch.save({"model": model.state_dict(), "optimizer_step": step, "config": config.__dict__}, output / f"step-{step}.pt")
    return history
