"""Leakage-free construction of native and hourly PV grids."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import IntervalSemantics, PVSeries, TimestampLabel


def _require_admitted_time_metadata(series: PVSeries) -> None:
    if series.timezone is None:
        raise ValueError("timezone is unknown; dependent real-data preprocessing is forbidden")
    if series.interval_semantics is None:
        raise ValueError("interval semantics are unknown; hourly aggregation rule cannot be chosen")
    if series.timestamp_label is None:
        raise ValueError("timestamp label semantics are unknown")


def build_regular_native_grid(series: PVSeries) -> PVSeries:
    """Insert missing native timestamps as NaN without imputing any target."""
    if not len(series.timestamp):
        return series
    frequency = pd.Timedelta(minutes=series.native_interval_minutes)
    grid = pd.date_range(series.timestamp[0], series.timestamp[-1], freq=frequency)
    indexed = pd.Series(series.target, index=series.timestamp).reindex(grid)
    values = indexed.to_numpy(dtype=np.float64)
    return PVSeries(
        dataset_id=series.dataset_id,
        physical_site_id=series.physical_site_id,
        physical_location_group_id=series.physical_location_group_id,
        timestamp=grid,
        target=values,
        observed_mask=np.isfinite(values),
        unit=series.unit,
        timezone=series.timezone,
        interval_semantics=series.interval_semantics,
        timestamp_label=series.timestamp_label,
        native_interval_minutes=series.native_interval_minutes,
        source_path=series.source_path,
        source_sha256=series.source_sha256,
    )


def aggregate_to_hourly(series: PVSeries) -> PVSeries:
    """Aggregate complete high-frequency intervals to right-labelled hourly average power."""
    _require_admitted_time_metadata(series)
    if 60 % series.native_interval_minutes:
        raise ValueError("native interval must divide one hour exactly")
    if series.native_interval_minutes > 60:
        raise ValueError("lower-frequency data cannot be upsampled to hourly")

    regular = build_regular_native_grid(series)
    interval = pd.Timedelta(minutes=regular.native_interval_minutes)
    labels = regular.timestamp
    if regular.timestamp_label is TimestampLabel.LEFT:
        labels = labels + interval

    frame = pd.DataFrame(
        {"target": regular.target, "observed": regular.observed_mask.astype(np.int8)},
        index=labels,
    )
    expected = 60 // regular.native_interval_minutes
    grouper = pd.Grouper(freq="1h", label="right", closed="right")
    observed_count = frame["observed"].groupby(grouper).sum()

    if regular.interval_semantics is IntervalSemantics.AVERAGE_POWER:
        weighted = frame["target"] * regular.native_interval_minutes
        hourly = weighted.groupby(grouper).sum(min_count=1) / 60.0
        output_unit = regular.unit
    elif regular.interval_semantics is IntervalSemantics.INTERVAL_ENERGY:
        # A complete bin represents exactly one hour, so summed energy / 1 h is power.
        hourly = frame["target"].groupby(grouper).sum(min_count=1)
        output_unit = _energy_to_power_unit(regular.unit)
    else:  # pragma: no cover - enum plus admission check make this defensive.
        raise AssertionError(regular.interval_semantics)

    complete = observed_count.eq(expected)
    hourly = hourly.where(complete)
    values = hourly.to_numpy(dtype=np.float64)
    return PVSeries(
        dataset_id=regular.dataset_id,
        physical_site_id=regular.physical_site_id,
        physical_location_group_id=regular.physical_location_group_id,
        timestamp=pd.DatetimeIndex(hourly.index),
        target=values,
        observed_mask=np.isfinite(values),
        unit=output_unit,
        timezone="UTC",
        interval_semantics=IntervalSemantics.AVERAGE_POWER,
        timestamp_label=TimestampLabel.RIGHT,
        native_interval_minutes=60,
        source_path=regular.source_path,
        source_sha256=regular.source_sha256,
    )


def _energy_to_power_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    mapping = {"Wh": "W", "kWh": "kW", "MWh": "MW"}
    if unit not in mapping:
        raise ValueError(f"unsupported interval-energy unit: {unit}")
    return mapping[unit]
