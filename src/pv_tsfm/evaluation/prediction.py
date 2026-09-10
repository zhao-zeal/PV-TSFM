"""Strict inference over a locked origin manifest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_tsfm.data.manifests import validate_origin_manifest
from pv_tsfm.data.schema import PVSeries
from pv_tsfm.models.base import Forecaster


def predict_manifest_rows(
    model: Forecaster,
    series: PVSeries,
    manifest: pd.DataFrame,
    *,
    run_id: str,
    seed: int | None,
) -> pd.DataFrame:
    validate_origin_manifest(manifest)
    eligible = manifest.loc[
        manifest["eligible"]
        & manifest["dataset"].eq(series.dataset_id)
        & manifest["site"].eq(series.physical_site_id)
    ].copy()
    position = {timestamp: index for index, timestamp in enumerate(series.timestamp)}
    rows = []
    for record in eligible.itertuples(index=False):
        origin = pd.Timestamp(record.origin)
        origin_position = position[origin]
        context_slice = slice(origin_position - 335, origin_position + 1)
        forecast = model.predict(
            series.target[context_slice],
            series.observed_mask[context_slice],
            horizon=24,
        )
        for lead, raw in enumerate(forecast, start=1):
            target_position = origin_position + lead
            rows.append({
                "run_id": run_id,
                "method": model.method,
                "seed": seed,
                "dataset": series.dataset_id,
                "site": series.physical_site_id,
                "origin": origin,
                "target_time": series.timestamp[target_position],
                "lead": lead,
                "y_true": series.target[target_position],
                "q": 0.5,
                "forecast_raw": float(raw),
                "forecast_post": max(0.0, float(raw)),
            })
    return pd.DataFrame(rows)
