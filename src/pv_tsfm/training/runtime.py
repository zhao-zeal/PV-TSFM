"""GPU visibility enforcement and auditable runtime metadata."""

from __future__ import annotations

import os


def visible_physical_gpu_ids() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return []
    try:
        ids = [int(item.strip()) for item in raw.split(",")]
    except ValueError as error:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain explicit numeric physical IDs") from error
    if len(ids) != len(set(ids)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate GPU IDs")
    if len(ids) > 2 or any(gpu not in {0, 1} for gpu in ids):
        raise RuntimeError("PV-TSFM may expose only physical GPUs 0 and/or 1, at most two total")
    return ids


def validate_run_gpu_policy(*, method: str, requested_gpu_count: int, single_gpu_oom_recorded: bool = False) -> list[int]:
    visible = visible_physical_gpu_ids()
    if len(visible) != requested_gpu_count:
        raise RuntimeError(f"requested {requested_gpu_count} GPUs but CUDA_VISIBLE_DEVICES exposes {visible}")
    if requested_gpu_count == 2 and not (method == "E5_full_cpt" and single_gpu_oom_recorded):
        raise RuntimeError("two-GPU single-run mode is reserved for Full CPT after a recorded legal single-GPU OOM")
    if requested_gpu_count not in {1, 2}:
        raise RuntimeError("training runs require one GPU, or conditionally two for Full CPT")
    return visible


def gpu_metadata(physical_ids: list[int]) -> dict:
    import torch

    if torch.cuda.device_count() != len(physical_ids):
        raise RuntimeError("PyTorch visible GPU count does not match CUDA_VISIBLE_DEVICES")
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_ids": physical_ids,
        "gpu_count": len(physical_ids),
        "gpus": [
            {
                "logical_id": index,
                "physical_id": physical_ids[index],
                "model": torch.cuda.get_device_name(index),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(index),
            }
            for index in range(len(physical_ids))
        ],
    }
