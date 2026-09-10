import numpy as np
import pandas as pd
import pytest

from pv_tsfm.data import (
    IntervalSemantics,
    PVSeries,
    TimestampLabel,
    aggregate_to_hourly,
    build_regular_native_grid,
)


def series(values, semantics, unit, *, label=TimestampLabel.RIGHT):
    target = np.asarray(values, dtype=float)
    return PVSeries(
        dataset_id="synthetic",
        physical_site_id="site-0",
        physical_location_group_id="location-0",
        timestamp=pd.date_range("2020-01-01 00:15", periods=len(target), freq="15min", tz="UTC"),
        target=target,
        observed_mask=np.isfinite(target),
        unit=unit,
        timezone="UTC",
        interval_semantics=semantics,
        timestamp_label=label,
        native_interval_minutes=15,
    )


def test_average_power_and_interval_energy_match():
    power = aggregate_to_hourly(series([2.0] * 8, IntervalSemantics.AVERAGE_POWER, "kW"))
    energy = aggregate_to_hourly(series([0.5] * 8, IntervalSemantics.INTERVAL_ENERGY, "kWh"))
    np.testing.assert_array_equal(power.observed_mask, [True, True])
    np.testing.assert_allclose(power.target, [2.0, 2.0])
    np.testing.assert_allclose(energy.target, power.target)
    assert energy.unit == "kW"


def test_incomplete_hour_is_not_scored_or_filled():
    raw = series([2.0, np.nan, 2.0, 2.0], IntervalSemantics.AVERAGE_POWER, "kW")
    hourly = aggregate_to_hourly(raw)
    assert hourly.observed_mask.tolist() == [False]
    assert np.isnan(hourly.target[0])


def test_regular_grid_preserves_gap_as_unobserved():
    raw = series([1.0, 2.0, 3.0], IntervalSemantics.AVERAGE_POWER, "kW")
    raw = PVSeries(**{**raw.__dict__, "timestamp": raw.timestamp.delete(1), "target": raw.target[[0, 2]], "observed_mask": raw.observed_mask[[0, 2]]})
    regular = build_regular_native_grid(raw)
    np.testing.assert_allclose(regular.target, [1.0, np.nan, 3.0], equal_nan=True)
    assert regular.observed_mask.tolist() == [True, False, True]


def test_unknown_metadata_fails_dependent_hourly_run():
    raw = series([1.0] * 4, IntervalSemantics.AVERAGE_POWER, "kW")
    unknown = PVSeries(**{**raw.__dict__, "timezone": None})
    with pytest.raises(ValueError, match="timezone is unknown"):
        aggregate_to_hourly(unknown)


def test_model_inputs_are_power_and_mask_only():
    raw = series([1.0] * 4, IntervalSemantics.AVERAGE_POWER, "kW")
    assert set(raw.model_inputs()) == {"past_target", "past_observed_mask"}
