"""Validated frozen optimizer settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    seed: int
    training_steps: int = 1000
    schedule_steps: int = 1000
    microbatch: int = 32
    gradient_accumulation: int = 4
    effective_batch: int = 128
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_fraction: float = 0.05
    context: int = 336
    horizon: int = 24

    @property
    def learning_rate(self) -> float:
        return {"E4_shared_lora_cpt": 1e-5, "E5_full_cpt": 1e-6}[self.method]

    def __post_init__(self) -> None:
        if self.method not in {"E4_shared_lora_cpt", "E5_full_cpt"}:
            raise ValueError(f"unsupported MVP training method: {self.method}")
        if self.seed not in {11, 22, 33}:
            raise ValueError("MVP seed must be one of 11, 22, 33")
        if self.training_steps not in {100, 1000}:
            raise ValueError("training_steps must be 100 profiling steps or 1000 formal steps")
        if self.schedule_steps != 1000:
            raise ValueError("scheduler length remains 1000 even for profiling/selected refits")
        if self.microbatch * self.gradient_accumulation != self.effective_batch:
            raise ValueError("microbatch * gradient accumulation must equal effective batch 128")
        if self.effective_batch != 128:
            raise ValueError("effective batch is frozen at 128")
        if (self.weight_decay, self.gradient_clip, self.warmup_fraction, self.context, self.horizon) != (
            0.01, 1.0, 0.05, 336, 24
        ):
            raise ValueError("training protocol constants were changed")
