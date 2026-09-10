#!/usr/bin/env python3
"""Run E0 and E1 on the locked MMSP published-series exploratory origins."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pv_tsfm.models.chronos2 import MODEL_ID, MODEL_REVISION
from pv_tsfm.training.runtime import gpu_metadata, validate_run_gpu_policy


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


def make_forecast(method: str, contexts: np.ndarray) -> np.ndarray:
    if method == "E0_last_value":
        return np.repeat(contexts[:, -1:], 24, axis=1)
    if method == "E0_daily_seasonal_persistence":
        return contexts[:, -24:].copy()
    if method == "E0_past_7_days_same_hour_mean":
        return contexts[:, -168:].reshape(len(contexts), 7, 24).mean(axis=1)
    raise ValueError(method)


def metric_rows(
    *, method: str, site: str, truth: np.ndarray, forecast: np.ndarray
) -> list[dict]:
    rows = []
    for horizon in (1, 4, 24):
        y = truth[:, :horizon].reshape(-1)
        for kind, pred in (
            ("raw", forecast[:, :horizon].reshape(-1)),
            ("post", np.maximum(0.0, forecast[:, :horizon]).reshape(-1)),
        ):
            error = pred - y
            rows.append(
                {
                    "method": method,
                    "dataset": "MMSP_published_series",
                    "site": site,
                    "horizon_steps": horizon,
                    "forecast_kind": kind,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "sum_absolute_error": float(np.abs(error).sum()),
                    "sum_squared_error": float(np.square(error).sum()),
                    "points": len(error),
                    "origins": len(truth),
                }
            )
    return rows


def write_prediction_chunk(
    path: Path,
    *,
    method: str,
    site: str,
    origins: pd.DataFrame,
    raw_timestamps: np.ndarray,
    truth: np.ndarray,
    forecast: np.ndarray,
    first: bool,
) -> None:
    n = len(origins)
    origin_index = origins.origin_index.to_numpy(int)
    target_index = origin_index[:, None] + np.arange(1, 25)[None, :]
    frame = pd.DataFrame(
        {
            "track_id": "mmsp-published-series-exploratory-v1",
            "run_id": "mmsp-published-e01",
            "method": method,
            "seed": "",
            "dataset": "MMSP_published_series",
            "site": site,
            "origin_index": np.repeat(origin_index, 24),
            "origin_raw_timestamp": np.repeat(
                raw_timestamps[origin_index], 24
            ),
            "target_index": target_index.reshape(-1),
            "target_raw_timestamp": raw_timestamps[target_index.reshape(-1)],
            "lead": np.tile(np.arange(1, 25), n),
            "y_true": truth.reshape(-1),
            "q": 0.5,
            "forecast_raw": forecast.reshape(-1),
            "forecast_post": np.maximum(0.0, forecast).reshape(-1),
            "timezone": "unknown",
            "interval_semantics": "unknown",
        }
    )
    frame.to_csv(
        path,
        index=False,
        mode="w" if first else "a",
        header=first,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/home/zhaopp/workspace/FusionSF/data/MMSP/data/solar_power/solar_power.csv"
        ),
    )
    parser.add_argument(
        "--origins",
        type=Path,
        default=Path(
            "manifests/mmsp_published_series_exploratory/origins/test_origins.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mmsp_published_series_exploratory/e01"),
    )
    parser.add_argument("--inference-batch-size", type=int, default=512)
    args = parser.parse_args()
    started = time.time()

    physical_ids = validate_run_gpu_policy(
        method="E1_original_chronos2", requested_gpu_count=1
    )
    import torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import snapshot_download

    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    origins = pd.read_csv(args.origins, dtype={"site": str})
    origins = origins.loc[origins.eligible].copy()
    if origins.groupby("site").size().nunique() != 1 or len(origins) != 30_540:
        raise RuntimeError("locked test origins are not 2,545 per site / 30,540 total")
    data = pd.read_csv(args.data, usecols=["datetime", "power", "site"])
    data["site"] = data.site.astype(str)

    model_path = snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    pipeline = Chronos2Pipeline.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        local_files_only=True,
    )

    metrics = []
    method_paths = {
        method: predictions_dir / f"{method}.csv.gz" for method in METHODS
    }
    first_write = {method: True for method in METHODS}
    for site in sorted(origins.site.unique(), key=int):
        site_data = data.loc[data.site.eq(site)].reset_index(drop=True)
        target = site_data.power.to_numpy(np.float32)
        raw_timestamps = site_data.datetime.astype(str).to_numpy()
        site_origins = origins.loc[origins.site.eq(site)].sort_values("origin_index")
        origin_index = site_origins.origin_index.to_numpy(int)
        contexts = np.stack(
            [target[index - 335 : index + 1] for index in origin_index]
        )
        truth = np.stack([target[index + 1 : index + 25] for index in origin_index])
        if not np.isfinite(contexts).all() or not np.isfinite(truth).all():
            raise RuntimeError(f"non-finite locked context/target for site {site}")

        for method in METHODS[:3]:
            forecast = make_forecast(method, contexts)
            metrics.extend(
                metric_rows(method=method, site=site, truth=truth, forecast=forecast)
            )
            write_prediction_chunk(
                method_paths[method],
                method=method,
                site=site,
                origins=site_origins,
                raw_timestamps=raw_timestamps,
                truth=truth,
                forecast=forecast,
                first=first_write[method],
            )
            first_write[method] = False

        forecast_parts = []
        for start in range(0, len(contexts), args.inference_batch_size):
            inputs = contexts[start : start + args.inference_batch_size, None, :]
            _, medians = pipeline.predict_quantiles(
                inputs,
                prediction_length=24,
                quantile_levels=[0.5],
                batch_size=args.inference_batch_size,
                context_length=336,
                cross_learning=False,
            )
            forecast_parts.append(
                np.stack(
                    [np.asarray(item, dtype=np.float32).reshape(24) for item in medians]
                )
            )
        forecast = np.concatenate(forecast_parts)
        if forecast.shape != truth.shape or not np.isfinite(forecast).all():
            raise RuntimeError(f"invalid E1 forecasts for site {site}: {forecast.shape}")
        method = METHODS[3]
        metrics.extend(
            metric_rows(method=method, site=site, truth=truth, forecast=forecast)
        )
        write_prediction_chunk(
            method_paths[method],
            method=method,
            site=site,
            origins=site_origins,
            raw_timestamps=raw_timestamps,
            truth=truth,
            forecast=forecast,
            first=first_write[method],
        )
        first_write[method] = False
        print(f"completed site {site}: {len(site_origins)} shared origins", flush=True)

    by_site = pd.DataFrame(metrics)
    by_site.to_csv(args.output_dir / "metrics_by_site.csv", index=False)
    summary_rows = []
    for keys, group in by_site.groupby(
        ["method", "horizon_steps", "forecast_kind"], sort=True
    ):
        method, horizon, kind = keys
        total_points = int(group.points.sum())
        summary_rows.append(
            {
                "method": method,
                "horizon_steps": horizon,
                "forecast_kind": kind,
                "macro_site_mae": float(group.mae.mean()),
                "macro_site_rmse": float(group.rmse.mean()),
                "micro_mae": float(group.sum_absolute_error.sum() / total_points),
                "micro_rmse": float(
                    math.sqrt(group.sum_squared_error.sum() / total_points)
                ),
                "sites": int(group.site.nunique()),
                "origins": int(group.origins.sum()),
                "points": total_points,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    runtime = {
        "track_id": "mmsp-published-series-exploratory-v1",
        "status": "complete",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "test_origins": len(origins),
        "origins_per_site": origins.groupby("site").size().astype(int).to_dict(),
        "methods": list(METHODS),
        "elapsed_seconds": time.time() - started,
        "gpu": gpu_metadata(physical_ids),
        "origin_manifest": {
            "path": str(args.origins.resolve()),
            "sha256": sha256(args.origins),
        },
        "prediction_files": {
            method: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "rows": 30_540 * 24,
            }
            for method, path in method_paths.items()
        },
        "claim_limit": "nominal-hourly published sequence; not verified real hourly-average power",
    }
    (args.output_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
