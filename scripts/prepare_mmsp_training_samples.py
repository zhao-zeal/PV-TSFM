#!/usr/bin/env python3
"""Freeze the shared E4/E5 MMSP window stream for seeds 11/22/33."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pv_tsfm.training.sampling import build_window_sampling_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    train_path = Path(
        "manifests/mmsp_published_series_exploratory/origins/train_origins.csv.gz"
    )
    output_dir = Path("manifests/mmsp_published_series_exploratory/training_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(
        train_path,
        usecols=[
            "original_source_id",
            "physical_location_id",
            "physical_site_id",
            "window_id",
        ],
        dtype={"physical_location_id": str, "physical_site_id": str},
    )
    if len(windows) != 634_432 or windows.window_id.duplicated().any():
        raise RuntimeError("training-window source manifest invariant failed")
    artifacts = {}
    for seed in (11, 22, 33):
        samples = build_window_sampling_manifest(
            windows, seed=seed, optimizer_steps=1000, effective_batch=128
        )
        path = output_dir / f"seed{seed}_steps1000_batch128.csv.gz"
        samples.to_csv(
            path,
            index=False,
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        )
        artifacts[str(seed)] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": len(samples),
            "optimizer_steps": int(samples.optimizer_step.nunique()),
            "batch_positions_per_step": int(
                samples.groupby("optimizer_step").size().unique().item()
            ),
        }
    manifest = {
        "track_id": "mmsp-published-series-exploratory-v1",
        "source_train_windows": 634432,
        "source_train_manifest_sha256": sha256(train_path),
        "effective_batch": 128,
        "formal_optimizer_steps": 1000,
        "same_seed_stream_across_E4_E5": True,
        "artifacts": artifacts,
    }
    output = output_dir / "sampling_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
