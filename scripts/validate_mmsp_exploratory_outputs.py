#!/usr/bin/env python3
"""Validate and summarize completed MMSP exploratory artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRACK_OUTPUT = ROOT / "outputs/mmsp_published_series_exploratory"
REPORT_DIR = ROOT / "reports/results/mmsp_published_series_exploratory"
METHODS = (
    "E0_last_value",
    "E0_daily_seasonal_persistence",
    "E0_past_7_days_same_hour_mean",
    "E1_original_chronos2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_dir = TRACK_OUTPUT / "e01/predictions"
    key_columns = ["site", "origin_index", "target_index", "lead", "y_true"]
    reference = None
    prediction_audit = {}
    for method in METHODS:
        path = prediction_dir / f"{method}.csv.gz"
        frame = pd.read_csv(path, usecols=["method", *key_columns], dtype={"site": str})
        if len(frame) != 732_960 or frame.method.unique().tolist() != [method]:
            raise RuntimeError(f"prediction row/method invariant failed for {method}")
        keys = frame[key_columns]
        if keys.duplicated(["site", "origin_index", "lead"]).any():
            raise RuntimeError(f"duplicate prediction keys for {method}")
        if reference is None:
            reference = keys
        elif not keys.equals(reference):
            raise RuntimeError(f"origin/target keys differ for {method}")
        prediction_audit[method] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": len(frame),
            "sites": int(frame.site.nunique()),
            "origins": int(frame.groupby(["site", "origin_index"]).ngroups),
            "leads": sorted(frame.lead.unique().astype(int).tolist()),
        }

    training_rows = []
    for method, stem in (
        ("E4_shared_lora_cpt", "e4"),
        ("E5_full_cpt", "e5"),
    ):
        for seed in (11, 22, 33):
            run_dir = TRACK_OUTPUT / f"training/{stem}_seed{seed}"
            runtime_path = run_dir / "runtime.json"
            history_path = run_dir / "history.csv"
            runtime = json.loads(runtime_path.read_text())
            history = pd.read_csv(history_path)
            if runtime["status"] != "complete" or len(history) != 1000:
                raise RuntimeError(f"incomplete training run: {method} seed {seed}")
            if history.step.tolist() != list(range(1, 1001)):
                raise RuntimeError(f"non-contiguous optimizer steps: {method} seed {seed}")
            if not np.isfinite(history[["loss", "lr", "grad_norm"]].to_numpy()).all():
                raise RuntimeError(f"non-finite training history: {method} seed {seed}")
            checkpoints = {}
            for step in (500, 1000):
                checkpoint = run_dir / f"checkpoints/step-{step}.pt"
                if not checkpoint.is_file():
                    raise RuntimeError(f"missing checkpoint: {checkpoint}")
                checkpoints[str(step)] = {
                    "path": str(checkpoint.resolve()),
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": sha256(checkpoint),
                }
            training_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "status": runtime["status"],
                    "optimizer_steps": len(history),
                    "learning_rate": runtime["learning_rate"],
                    "microbatch": runtime["microbatch"],
                    "gradient_accumulation": runtime["gradient_accumulation"],
                    "effective_batch": runtime["effective_batch"],
                    "physical_gpu_id": runtime["gpu"]["gpus"][0]["physical_id"],
                    "peak_vram_bytes": runtime["gpu"]["gpus"][0]["peak_vram_bytes"],
                    "elapsed_seconds": runtime["elapsed_seconds"],
                    "first_loss": history.loss.iloc[0],
                    "final_loss": history.loss.iloc[-1],
                    "runtime_path": str(runtime_path.resolve()),
                    "history_sha256": sha256(history_path),
                    "checkpoints": checkpoints,
                }
            )

    e01_metrics = TRACK_OUTPUT / "e01/metrics_summary.csv"
    by_site_metrics = TRACK_OUTPUT / "e01/metrics_by_site.csv"
    shutil.copyfile(e01_metrics, REPORT_DIR / "e01_metrics_summary.csv")
    shutil.copyfile(by_site_metrics, REPORT_DIR / "e01_metrics_by_site.csv")
    training_summary = pd.DataFrame(training_rows)
    compact = training_summary.drop(columns="checkpoints")
    compact.to_csv(REPORT_DIR / "training_summary.csv", index=False)
    audit = {
        "track_id": "mmsp-published-series-exploratory-v1",
        "status": "complete_through_E4_E5_training_not_adapted_model_scoring",
        "shared_prediction_keys_verified": True,
        "test_origins": 30540,
        "prediction_artifacts": prediction_audit,
        "profiling": {
            path.parent.name: json.loads(path.read_text())
            for path in sorted((TRACK_OUTPUT / "profiles").glob("*/runtime.json"))
        },
        "training": training_rows,
        "result_tables": {
            "e01_summary": str((REPORT_DIR / "e01_metrics_summary.csv").resolve()),
            "e01_by_site": str((REPORT_DIR / "e01_metrics_by_site.csv").resolve()),
            "training_summary": str((REPORT_DIR / "training_summary.csv").resolve()),
        },
        "claim_limit": "nominal-hourly published sequence; not verified real hourly-average power",
        "does_not_replace": "StateGrid-to-PVOD or formal two-task MVP",
    }
    output = REPORT_DIR / "execution_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": audit["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
