import numpy as np
import pandas as pd

from pv_tsfm.data.manifests import build_origin_manifest
from pv_tsfm.data.schema import IntervalSemantics, PVSeries, TimestampLabel
from pv_tsfm.evaluation import predict_manifest_rows
from pv_tsfm.models.chronos2 import Chronos2Forecaster


class FakeChronosPipeline:
    quantiles = [0.1, 0.5, 0.9]

    def predict_quantiles(self, inputs, **kwargs):
        assert set(kwargs) == {
            "prediction_length", "quantile_levels", "batch_size", "context_length", "cross_learning"
        }
        context = np.asarray(inputs[0])
        median = np.repeat(context[-1], kwargs["prediction_length"])[None, :]
        return [median[..., None]], [median]


def make_series(target):
    target = np.asarray(target, dtype=float)
    return PVSeries(
        dataset_id="synthetic",
        physical_site_id="site",
        physical_location_group_id="group",
        timestamp=pd.date_range("2020-01-01", periods=len(target), freq="1h", tz="UTC"),
        target=target,
        observed_mask=np.isfinite(target),
        unit="kW",
        timezone="UTC",
        interval_semantics=IntervalSemantics.AVERAGE_POWER,
        timestamp_label=TimestampLabel.RIGHT,
        native_interval_minutes=60,
    )


def test_changing_future_truth_does_not_change_prediction():
    original = make_series(np.arange(500, dtype=float))
    manifest = build_origin_manifest(original, group_role="validation")
    one_origin = manifest.loc[manifest.eligible].iloc[[0]].copy()
    origin_position = original.timestamp.get_loc(one_origin.iloc[0].origin)
    changed_target = original.target.copy()
    changed_target[origin_position + 1 : origin_position + 25] += 1_000_000
    changed = make_series(changed_target)
    model = Chronos2Forecaster(FakeChronosPipeline())
    first = predict_manifest_rows(model, original, one_origin, run_id="first", seed=None)
    second = predict_manifest_rows(model, changed, one_origin, run_id="second", seed=None)
    np.testing.assert_array_equal(first.forecast_raw, second.forecast_raw)
    assert not np.array_equal(first.y_true, second.y_true)
