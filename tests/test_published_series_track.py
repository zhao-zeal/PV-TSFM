import numpy as np
import pandas as pd

from pv_tsfm.data.published_series import (
    build_published_series_origins,
    validate_published_series_origins,
)


def test_mmsp_expected_complete_held_out_origins_without_timezone_inference():
    n = 12_840
    labels = pd.Series(pd.date_range("2021-01-01", periods=n, freq="h").astype(str))
    frame = build_published_series_origins(
        site_id="68",
        raw_timestamps=labels,
        observed_mask=np.ones(n, dtype=bool),
        group_role="test",
    )
    eligible = frame.loc[frame.eligible]
    assert len(eligible) == 2_545
    assert eligible.origin_index.iloc[0] == 10_271
    assert eligible.origin_index.iloc[-1] == 12_815
    assert eligible.time_boundary_index.eq(10_272).all()
    assert eligible.timezone.eq("unknown").all()
    assert eligible.interval_semantics.eq("unknown").all()
    validate_published_series_origins(frame)


def test_mmsp_expected_complete_training_windows():
    n = 12_840
    frame = build_published_series_origins(
        site_id="0",
        raw_timestamps=pd.Series([f"label-{i}" for i in range(n)]),
        observed_mask=np.ones(n, dtype=bool),
        group_role="train",
    )
    eligible = frame.loc[frame.eligible]
    assert len(eligible) == 9_913
    assert eligible.origin_index.iloc[0] == 335
    assert eligible.origin_index.iloc[-1] == 10_247


def test_missing_future_is_not_eligible_or_filled():
    n = 12_840
    mask = np.ones(n, dtype=bool)
    mask[12_839] = False
    frame = build_published_series_origins(
        site_id="68",
        raw_timestamps=pd.Series([f"label-{i}" for i in range(n)]),
        observed_mask=mask,
        group_role="test",
    )
    assert len(frame.loc[frame.eligible]) == 2_544
    assert frame.loc[frame.origin_index.eq(12_815), "exclude_reason"].item() == "incomplete_future_target"
