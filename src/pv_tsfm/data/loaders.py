"""Dataset-specific raw readers; weather and other covariates are never returned."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .schema import IntervalSemantics, PVSeries, TimestampLabel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_time(values: pd.Series, timezone: str | None) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if timezone is None:
        return parsed
    if parsed.tz is None:
        parsed = parsed.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return parsed.tz_convert("UTC")


def _make_series(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    site_id: str,
    time_col: str,
    target_col: str,
    native_interval_minutes: int,
    timezone: str | None,
    unit: str | None,
    interval_semantics: IntervalSemantics | None,
    timestamp_label: TimestampLabel | None,
    physical_location_group_id: str | None,
    source_path: str,
    source_sha256: str,
) -> PVSeries:
    selected = frame[[time_col, target_col]].copy()
    selected[time_col] = _canonical_time(selected[time_col], timezone)
    selected[target_col] = pd.to_numeric(selected[target_col], errors="coerce")
    selected = selected.sort_values(time_col)
    if selected[time_col].duplicated().any():
        raise ValueError(f"duplicate timestamps for {dataset_id}/{site_id}")
    target = selected[target_col].to_numpy(dtype=np.float64)
    return PVSeries(
        dataset_id=dataset_id,
        physical_site_id=str(site_id),
        physical_location_group_id=physical_location_group_id,
        timestamp=pd.DatetimeIndex(selected[time_col]),
        target=target,
        observed_mask=np.isfinite(target),
        unit=unit,
        timezone=timezone,
        interval_semantics=interval_semantics,
        timestamp_label=timestamp_label,
        native_interval_minutes=native_interval_minutes,
        source_path=source_path,
        source_sha256=source_sha256,
    )


def load_mmsp(
    path: str | Path,
    *,
    timezone: str | None = None,
    unit: str | None = None,
    interval_semantics: IntervalSemantics | None = None,
    timestamp_label: TimestampLabel | None = None,
    location_groups: dict[str, str] | None = None,
) -> list[PVSeries]:
    path = Path(path)
    frame = pd.read_csv(path, usecols=["datetime", "power", "site"])
    digest = _sha256(path)
    result = []
    for site, part in frame.groupby("site", sort=True):
        site_id = str(site)
        result.append(_make_series(
            part, dataset_id="MMSP", site_id=site_id, time_col="datetime", target_col="power",
            native_interval_minutes=60, timezone=timezone, unit=unit,
            interval_semantics=interval_semantics, timestamp_label=timestamp_label,
            physical_location_group_id=None if location_groups is None else location_groups.get(site_id),
            source_path=str(path.resolve()), source_sha256=digest,
        ))
    return result


def load_stategrid(
    path: str | Path,
    *,
    site_id: str,
    timezone: str | None = None,
    interval_semantics: IntervalSemantics | None = None,
    timestamp_label: TimestampLabel | None = None,
    physical_location_group_id: str | None = None,
) -> PVSeries:
    path = Path(path)
    frame = pd.read_csv(path, usecols=["date", "Power (MW)"])
    return _make_series(
        frame, dataset_id="StateGrid", site_id=site_id, time_col="date", target_col="Power (MW)",
        native_interval_minutes=15, timezone=timezone, unit="MW",
        interval_semantics=interval_semantics, timestamp_label=timestamp_label,
        physical_location_group_id=physical_location_group_id,
        source_path=str(path.resolve()), source_sha256=_sha256(path),
    )


def load_ai_pvod_member(
    archive_path: str | Path,
    member: str,
    *,
    timezone: str | None = None,
    unit: str | None = None,
    interval_semantics: IntervalSemantics | None = None,
    timestamp_label: TimestampLabel | None = None,
    physical_location_group_id: str | None = None,
) -> PVSeries:
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read(member)
    frame = pd.read_csv(io.BytesIO(raw))
    locations = frame["Location"].dropna().astype(str).unique()
    if len(locations) != 1:
        raise ValueError(f"expected one Location in {member}, found {locations.tolist()}")
    target_col = "Active_Power" if member.endswith("KDASC.csv") else "power"
    dataset_id = "DKASC" if member.endswith("KDASC.csv") else "PVOD_China"
    return _make_series(
        frame, dataset_id=dataset_id, site_id=locations[0], time_col="Time", target_col=target_col,
        native_interval_minutes=15, timezone=timezone, unit=unit,
        interval_semantics=interval_semantics, timestamp_label=timestamp_label,
        physical_location_group_id=physical_location_group_id,
        source_path=f"{archive_path.resolve()}::{member}", source_sha256=hashlib.sha256(raw).hexdigest(),
    )
