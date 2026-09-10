#!/usr/bin/env python3
"""Profile or train E4/E5 on frozen MMSP published-series window streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pv_tsfm.models.chronos2 import MODEL_ID, MODEL_REVISION
from pv_tsfm.training.config import TrainingConfig
from pv_tsfm.training.runtime import gpu_metadata, validate_run_gpu_policy
from pv_tsfm.training.trainer import create_fresh_trainable_model, train_optimizer_steps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenBatches:
    def __init__(
        self,
        samples: pd.DataFrame,
        targets: dict[str, np.ndarray],
        *,
        microbatch: int,
        device: str,
    ) -> None:
        match = samples.window_id.str.extract(
            r"^MMSP\|site=(?P<site>\d+)\|origin_index=(?P<origin>\d+)$"
        )
        if match.isna().any().any():
            raise RuntimeError("invalid frozen window_id")
        self.sites = match.site.to_numpy()
        self.origins = match.origin.astype(int).to_numpy()
        self.targets = targets
        self.microbatch = microbatch
        self.device = device

    def __iter__(self):
        import torch

        for start in range(0, len(self.origins), self.microbatch):
            end = start + self.microbatch
            origins = self.origins[start:end]
            sites = self.sites[start:end]
            if len(origins) != self.microbatch:
                raise RuntimeError("frozen sample rows do not divide into microbatches")
            context = np.stack(
                [self.targets[site][origin - 335 : origin + 1] for site, origin in zip(sites, origins)]
            )
            future = np.stack(
                [self.targets[site][origin + 1 : origin + 25] for site, origin in zip(sites, origins)]
            )
            if not np.isfinite(context).all() or not np.isfinite(future).all():
                raise RuntimeError("frozen training stream contains missing target values")
            context_tensor = torch.as_tensor(context, dtype=torch.float32, device=self.device)
            future_tensor = torch.as_tensor(future, dtype=torch.float32, device=self.device)
            yield {
                "context": context_tensor,
                "context_mask": torch.ones_like(context_tensor, dtype=torch.bool),
                "future_target": future_tensor,
                "future_target_mask": torch.ones_like(future_tensor, dtype=torch.bool),
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["E4_shared_lora_cpt", "E5_full_cpt"])
    parser.add_argument("--seed", required=True, type=int, choices=[11, 22, 33])
    parser.add_argument("--steps", required=True, type=int, choices=[100, 1000])
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/home/zhaopp/workspace/FusionSF/data/MMSP/data/solar_power/solar_power.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()

    physical_ids = validate_run_gpu_policy(method=args.method, requested_gpu_count=1)
    config = TrainingConfig(
        method=args.method,
        seed=args.seed,
        training_steps=args.steps,
        microbatch=args.microbatch,
        gradient_accumulation=args.gradient_accumulation,
    )
    sample_path = Path(
        f"manifests/mmsp_published_series_exploratory/training_samples/seed{args.seed}_steps1000_batch128.csv.gz"
    )
    samples = pd.read_csv(sample_path, dtype={"physical_site_id": str})
    samples = samples.loc[samples.optimizer_step.le(args.steps)].copy()
    if len(samples) != args.steps * 128:
        raise RuntimeError("frozen sample manifest does not match optimizer-step budget")

    data = pd.read_csv(args.data, usecols=["power", "site"])
    data["site"] = data.site.astype(str)
    targets = {
        site: part.power.to_numpy(np.float32)
        for site, part in data.groupby("site", sort=True)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    torch.cuda.reset_peak_memory_stats()
    model, identity, base_hash = create_fresh_trainable_model(config, device="cuda")
    batches = FrozenBatches(
        samples,
        targets,
        microbatch=config.microbatch,
        device="cuda",
    )
    history = train_optimizer_steps(
        model, batches, config, output_dir=args.output_dir / "checkpoints"
    )
    history_path = args.output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    runtime = {
        "track_id": "mmsp-published-series-exploratory-v1",
        "status": "complete",
        "run_kind": "profiling" if args.steps == 100 else "formal_exploratory_training",
        "method": args.method,
        "seed": args.seed,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "training_steps": args.steps,
        "schedule_steps": config.schedule_steps,
        "learning_rate": config.learning_rate,
        "microbatch": config.microbatch,
        "gradient_accumulation": config.gradient_accumulation,
        "effective_batch": config.effective_batch,
        "sample_rows": len(samples),
        "sample_manifest": str(sample_path.resolve()),
        "sample_manifest_sha256": sha256(sample_path),
        "elapsed_seconds": time.time() - started,
        "seconds_per_step": (time.time() - started) / args.steps,
        "gpu": gpu_metadata(physical_ids),
        "base_parameter_hash_before_training": base_hash,
        "identity": identity,
        "history": str(history_path.resolve()),
        "history_sha256": sha256(history_path),
        "claim_limit": "nominal-hourly published sequence; not verified real hourly-average power",
    }
    (args.output_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(runtime, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
