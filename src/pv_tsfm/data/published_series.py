"""Index-based manifests for the MMSP published-series exploratory track.

This module deliberately does not localize timestamps or infer interval semantics.
Raw timestamps are retained as labels while all window arithmetic uses row positions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


CONTEXT_RECORDS = 336
HORIZON_RECORDS = 24


def build_published_series_origins(
    *,
    site_id: str,
    raw_timestamps: pd.Series,
    observed_mask: np.ndarray,
    group_role: str,
    context: int = CONTEXT_RECORDS,
    horizon: int = HORIZON_RECORDS,
) -> pd.DataFrame:
    """Enumerate complete index-based origins without assigning a timezone."""
    if group_role not in {"train", "validation", "test"}:
        raise ValueError(f"invalid group role: {group_role}")
    labels = raw_timestamps.astype(str).reset_index(drop=True)
    mask = np.asarray(observed_mask, dtype=bool)
    if len(labels) != len(mask):
        raise ValueError("timestamps and observed mask lengths differ")
    if context <= 0 or horizon <= 0:
        raise ValueError("context and horizon must be positive")
    n = len(labels)
    boundary = math.floor(0.8 * n)
    assigned_partition = "train" if group_role == "train" else "held_out"
    rows: list[dict] = []
    for origin_index in range(context - 1, n - horizon):
        target_start_index = origin_index + 1
        target_end_index = origin_index + horizon
        if target_end_index < boundary:
            label_partition = "train"
        elif target_start_index >= boundary:
            label_partition = "held_out"
        else:
            label_partition = "cross_boundary"
        reasons = []
        if label_partition != assigned_partition:
            reasons.append(
                "future_crosses_time_boundary"
                if label_partition == "cross_boundary"
                else "future_outside_assigned_partition"
            )
        if not mask[origin_index - context + 1 : origin_index + 1].all():
            reasons.append("incomplete_context")
        if not mask[target_start_index : target_end_index + 1].all():
            reasons.append("incomplete_future_target")
        rows.append(
            {
                "track_id": "mmsp-published-series-exploratory-v1",
                "dataset": "MMSP_published_series",
                "site": str(site_id),
                "group_role": group_role,
                "time_basis": "raw_ordered_timestamp_and_row_index",
                "timezone": "unknown",
                "interval_semantics": "unknown",
                "origin_index": origin_index,
                "context_start_index": origin_index - context + 1,
                "context_end_index": origin_index,
                "target_start_index": target_start_index,
                "target_end_index": target_end_index,
                "origin_raw_timestamp": labels.iloc[origin_index],
                "target_start_raw_timestamp": labels.iloc[target_start_index],
                "target_end_raw_timestamp": labels.iloc[target_end_index],
                "grid_length": n,
                "time_boundary_index": boundary,
                "label_partition": label_partition,
                "eligible": not reasons,
                "exclude_reason": None if not reasons else "|".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def validate_published_series_origins(
    frame: pd.DataFrame,
    *,
    context: int = CONTEXT_RECORDS,
    horizon: int = HORIZON_RECORDS,
) -> None:
    required = {
        "dataset",
        "site",
        "origin_index",
        "context_start_index",
        "context_end_index",
        "target_start_index",
        "target_end_index",
        "eligible",
        "exclude_reason",
        "timezone",
        "interval_semantics",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"published-series manifest missing columns: {sorted(missing)}")
    if frame.duplicated(["dataset", "site", "origin_index"]).any():
        raise ValueError("duplicate dataset/site/origin_index rows")
    eligible = frame.loc[frame.eligible]
    if eligible.exclude_reason.notna().any():
        raise ValueError("eligible origins cannot have exclusion reasons")
    if not eligible.timezone.eq("unknown").all():
        raise ValueError("exploratory timestamps must not be assigned a timezone")
    if not eligible.interval_semantics.eq("unknown").all():
        raise ValueError("exploratory interval semantics must remain unknown")
    if not (eligible.context_end_index == eligible.origin_index).all():
        raise ValueError("origin index must equal context end index")
    if not (
        eligible.context_end_index - eligible.context_start_index == context - 1
    ).all():
        raise ValueError("context record count mismatch")
    if not (eligible.target_start_index == eligible.origin_index + 1).all():
        raise ValueError("target must start at the next record")
    if not (
        eligible.target_end_index - eligible.target_start_index == horizon - 1
    ).all():
        raise ValueError("target record count mismatch")
