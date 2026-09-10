"""Frozen group splits and method-independent forecast-origin manifests."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .schema import PVSeries


SALT = "pv-tsfm-v1-20260909"
CONTEXT = 336
HORIZON = 24


def _group_hash(group_id: str, salt: str = SALT) -> str:
    return hashlib.sha256(f"{salt}|{group_id}".encode("utf-8")).hexdigest()


def assign_mmsp_group_roles(group_ids: Iterable[str], salt: str = SALT) -> pd.DataFrame:
    groups = sorted({str(group_id) for group_id in group_ids})
    if not groups or any(not group for group in groups):
        raise ValueError("canonical physical-location group IDs must be non-empty")
    ordered = sorted(groups, key=lambda group: (_group_hash(group, salt), group))
    if len(ordered) == 88:
        train_count, validation_count = 64, 12
        count_rule = "expected_88_groups"
    else:
        train_count = math.floor(0.72 * len(ordered))
        validation_count = math.floor(0.14 * len(ordered))
        count_rule = "fallback_72_14_14"
    roles = (["train"] * train_count + ["validation"] * validation_count
             + ["test"] * (len(ordered) - train_count - validation_count))
    return pd.DataFrame({
        "task": "mmsp_site_out",
        "canonical_physical_location_group_id": ordered,
        "sha256_sort_key": [_group_hash(group, salt) for group in ordered],
        "group_role": roles,
        "count_rule": count_rule,
    })


def assign_stategrid_roles(ordered_group_ids: Iterable[str]) -> pd.DataFrame:
    groups = [str(group) for group in ordered_group_ids]
    if len(groups) != 8 or len(set(groups)) != 8:
        raise ValueError("StateGrid task requires exactly eight verified independent ordered groups")
    return pd.DataFrame({
        "task": "stategrid_to_pvod",
        "canonical_physical_location_group_id": groups,
        "group_role": ["train"] * 6 + ["validation"] * 2,
        "count_rule": "protocol_source_counts_6_2",
    })


def validate_split_integrity(site_to_group: Mapping[str, str], split: pd.DataFrame) -> None:
    required = {"canonical_physical_location_group_id", "group_role"}
    if not required.issubset(split.columns):
        raise ValueError(f"split manifest missing columns: {sorted(required - set(split.columns))}")
    if split["canonical_physical_location_group_id"].duplicated().any():
        raise ValueError("one physical-location group appears in multiple split rows")
    role_by_group = split.set_index("canonical_physical_location_group_id")["group_role"]
    missing = sorted(set(site_to_group.values()) - set(role_by_group.index))
    if missing:
        raise ValueError(f"sites reference groups absent from split: {missing}")
    site_rows = pd.DataFrame({"site": list(site_to_group), "group": list(site_to_group.values())})
    site_rows["role"] = site_rows["group"].map(role_by_group)
    if site_rows.groupby("group")["role"].nunique().gt(1).any():
        raise ValueError("a physical-location group crosses split roles")


def build_origin_manifest(
    series: PVSeries,
    *,
    group_role: str,
    context: int = CONTEXT,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Enumerate candidate hourly origins and retain explicit exclusion reasons."""
    if series.native_interval_minutes != 60:
        raise ValueError("origin manifests require an hourly series")
    if series.timestamp.tz is None or str(series.timestamp.tz) != "UTC":
        raise ValueError("origin manifests require canonical UTC timestamps")
    if group_role not in {"train", "validation", "test"}:
        raise ValueError(f"invalid group role: {group_role}")
    if context <= 0 or horizon <= 0:
        raise ValueError("context and horizon must be positive")

    n = len(series.timestamp)
    boundary = math.floor(0.8 * n)
    rows = []
    for origin_position in range(context - 1, n - horizon):
        context_slice = slice(origin_position - context + 1, origin_position + 1)
        target_slice = slice(origin_position + 1, origin_position + horizon + 1)
        target_positions = np.arange(origin_position + 1, origin_position + horizon + 1)
        label_partition = "train" if np.all(target_positions < boundary) else (
            "held_out" if np.all(target_positions >= boundary) else "cross_boundary"
        )
        assigned_partition = "train" if group_role == "train" else "held_out"
        reasons = []
        if label_partition != assigned_partition:
            reasons.append(
                "future_crosses_time_boundary" if label_partition == "cross_boundary"
                else "future_outside_assigned_partition"
            )
        if not series.observed_mask[context_slice].all():
            reasons.append("incomplete_context")
        if not series.observed_mask[target_slice].all():
            reasons.append("incomplete_future_target")
        rows.append({
            "dataset": series.dataset_id,
            "site": series.physical_site_id,
            "physical_location_group_id": series.physical_location_group_id,
            "group_role": group_role,
            "origin": series.timestamp[origin_position],
            "context_start": series.timestamp[origin_position - context + 1],
            "context_end": series.timestamp[origin_position],
            "target_start": series.timestamp[origin_position + 1],
            "target_end": series.timestamp[origin_position + horizon],
            "grid_length": n,
            "time_boundary_position": boundary,
            "label_partition": label_partition,
            "eligible": not reasons,
            "exclude_reason": None if not reasons else "|".join(reasons),
        })
    return pd.DataFrame(rows)


def validate_origin_manifest(frame: pd.DataFrame, *, context: int = CONTEXT, horizon: int = HORIZON) -> None:
    required = {
        "dataset", "site", "origin", "context_start", "context_end", "target_start",
        "target_end", "eligible", "exclude_reason",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"origin manifest missing columns: {sorted(required - set(frame.columns))}")
    if frame.duplicated(["dataset", "site", "origin"]).any():
        raise ValueError("duplicate dataset/site/origin rows")
    eligible = frame.loc[frame["eligible"]].copy()
    if eligible["exclude_reason"].notna().any():
        raise ValueError("eligible origins cannot have exclusion reasons")
    for column in ["origin", "context_start", "context_end", "target_start", "target_end"]:
        eligible[column] = pd.to_datetime(eligible[column], utc=True)
    one_hour = pd.Timedelta(hours=1)
    if not (eligible["context_end"] == eligible["origin"]).all():
        raise ValueError("origin must equal context_end")
    if not (eligible["target_start"] == eligible["origin"] + one_hour).all():
        raise ValueError("target must start one hour after origin")
    if not (eligible["context_end"] - eligible["context_start"] == (context - 1) * one_hour).all():
        raise ValueError("context length mismatch")
    if not (eligible["target_end"] - eligible["target_start"] == (horizon - 1) * one_hour).all():
        raise ValueError("target horizon mismatch")
