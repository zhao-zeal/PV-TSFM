import numpy as np
import pandas as pd
import pytest

from pv_tsfm.evaluation.metrics import mae, rmse
from pv_tsfm.evaluation.scoring import aggregate_gains, compute_basic_scores, require_shared_predictions


def prediction_table(methods=("E1_original_chronos2", "E4_shared_lora_cpt")):
    rows = []
    for method in methods:
        for origin in pd.date_range("2020-01-01", periods=2, freq="1h", tz="UTC"):
            for lead in range(1, 25):
                truth = float(lead)
                rows.append({
                    "method": method,
                    "seed": 11,
                    "dataset": "d",
                    "site": "s",
                    "origin": origin,
                    "target_time": origin + pd.Timedelta(hours=lead),
                    "lead": lead,
                    "q": 0.5,
                    "y_true": truth,
                    "forecast_raw": truth + 1,
                    "forecast_post": truth + 1,
                })
    return pd.DataFrame(rows)


def test_metrics_match_hand_calculation():
    assert mae([0, 2], [1, 0]) == 1.5
    assert rmse([0, 2], [1, 0]) == pytest.approx(np.sqrt(2.5))


def test_identical_methods_have_zero_gain_at_all_locked_horizons():
    _, gains = compute_basic_scores(
        prediction_table(), methods=["E1_original_chronos2", "E4_shared_lora_cpt"]
    )
    adapted = gains.loc[gains.method.eq("E4_shared_lora_cpt")]
    assert set(adapted.horizon) == {1, 4, 24}
    np.testing.assert_allclose(adapted.mae_gain_percent, 0)
    np.testing.assert_allclose(adapted.rmse_gain_percent, 0)

    aggregated = aggregate_gains(
        prediction_table(), methods=["E1_original_chronos2", "E4_shared_lora_cpt"]
    )
    assert set(aggregated.aggregation) == {"site", "seed", "dataset"}
    assert set(aggregated.horizon) == {1, 4, 24}
    adapted = aggregated.loc[aggregated.method.eq("E4_shared_lora_cpt")]
    np.testing.assert_allclose(adapted.mae_gain_percent, 0)


def test_missing_one_origin_causes_primary_comparison_failure():
    table = prediction_table()
    remove = (
        table.method.eq("E4_shared_lora_cpt")
        & table.origin.eq(table.origin.min())
        & table.lead.eq(1)
    )
    with pytest.raises(ValueError, match="exactly the same"):
        require_shared_predictions(table.loc[~remove], ["E1_original_chronos2", "E4_shared_lora_cpt"])


def test_incomplete_path_is_rejected_even_when_both_methods_match():
    table = prediction_table()
    truncated = table.loc[~table.lead.eq(24)]
    with pytest.raises(ValueError, match="exactly leads 1..24"):
        compute_basic_scores(
            truncated, methods=["E1_original_chronos2", "E4_shared_lora_cpt"]
        )
