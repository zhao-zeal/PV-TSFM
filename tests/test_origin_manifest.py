import numpy as np
import pandas as pd
import pytest

from pv_tsfm.data.manifests import build_origin_manifest, validate_origin_manifest
from pv_tsfm.data.schema import IntervalSemantics, PVSeries, TimestampLabel


def make_series(n=500, missing=()):
    target = np.arange(n, dtype=float)
    target[list(missing)] = np.nan
    return PVSeries(
        dataset_id="synthetic",
        physical_site_id="site",
        physical_location_group_id="group",
        timestamp=pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
        target=target,
        observed_mask=np.isfinite(target),
        unit="kW",
        timezone="UTC",
        interval_semantics=IntervalSemantics.AVERAGE_POWER,
        timestamp_label=TimestampLabel.RIGHT,
        native_interval_minutes=60,
    )


def test_complete_future_must_stay_inside_partition():
    manifest = build_origin_manifest(make_series(), group_role="validation")
    eligible = manifest.loc[manifest.eligible]
    assert (eligible.target_start >= pd.Timestamp("2020-01-17 16:00", tz="UTC")).all()
    assert eligible.iloc[0].context_start < pd.Timestamp("2020-01-17 16:00", tz="UTC")
    validate_origin_manifest(manifest)


def test_missing_context_and_future_are_recorded():
    manifest = build_origin_manifest(make_series(missing=[350, 450]), group_role="validation")
    assert manifest.exclude_reason.fillna("").str.contains("incomplete_context").any()
    assert manifest.exclude_reason.fillna("").str.contains("incomplete_future_target").any()


def test_manifest_validation_rejects_duplicate_origin():
    manifest = build_origin_manifest(make_series(), group_role="test")
    duplicate = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_origin_manifest(duplicate)
