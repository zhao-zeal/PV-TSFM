"""Strict shared-origin scoring for E4 versus original Chronos-2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import mae, rmse


KEYS = ["seed", "dataset", "site", "origin", "target_time", "lead", "q"]
REQUIRED = set(KEYS + ["method", "y_true", "forecast_raw", "forecast_post"])


def require_shared_predictions(predictions: pd.DataFrame, methods: list[str]) -> None:
    missing = REQUIRED - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    selected = predictions.loc[predictions.method.isin(methods)].copy()
    absent = sorted(set(methods) - set(selected.method.unique()))
    if absent:
        raise ValueError(f"missing methods for comparison: {absent}")
    if selected.duplicated(["method", *KEYS]).any():
        raise ValueError("duplicate prediction keys within a method")
    reference = None
    reference_truth = None
    for method in methods:
        table = selected.loc[selected.method.eq(method)].sort_values(KEYS).reset_index(drop=True)
        keys = table[KEYS]
        truth = table[KEYS + ["y_true"]]
        if reference is None:
            reference, reference_truth = keys, truth
        elif not keys.equals(reference):
            raise ValueError("methods do not contain exactly the same origin/lead prediction keys")
        elif not np.array_equal(truth.y_true.to_numpy(), reference_truth.y_true.to_numpy(), equal_nan=True):
            raise ValueError("methods disagree on y_true for shared prediction keys")


def compute_basic_scores(
    predictions: pd.DataFrame,
    *,
    methods: list[str],
    reference_method: str = "E1_original_chronos2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_shared_predictions(predictions, methods)
    if reference_method not in methods:
        raise ValueError("reference method must be included")
    rows = []
    selected = predictions.loc[predictions.method.isin(methods) & predictions.q.eq(0.5)]
    lead_counts = selected.groupby(
        ["method", "seed", "dataset", "site", "origin"], dropna=False
    ).lead.agg(lambda values: set(values))
    expected_leads = set(range(1, 25))
    incomplete = lead_counts.loc[lead_counts.ne(expected_leads)]
    if not incomplete.empty:
        raise ValueError("every scored origin must contain exactly leads 1..24")
    for (method, seed, dataset, site), group in selected.groupby(
        ["method", "seed", "dataset", "site"], dropna=False, sort=True
    ):
        for horizon in (1, 4, 24):
            prefix = group.loc[group.lead.le(horizon)]
            if prefix.empty or prefix.lead.nunique() != horizon:
                raise ValueError(f"incomplete {horizon}h prefix for {method}/{seed}/{dataset}/{site}")
            for forecast_kind in ("raw", "post"):
                forecast = prefix[f"forecast_{forecast_kind}"]
                rows.append({
                    "method": method,
                    "seed": seed,
                    "dataset": dataset,
                    "site": site,
                    "horizon": horizon,
                    "forecast_kind": forecast_kind,
                    "mae": mae(prefix.y_true, forecast),
                    "rmse": rmse(prefix.y_true, forecast),
                    "points": len(prefix),
                    "origins": prefix.origin.nunique(),
                })
    scores = pd.DataFrame(rows)
    reference = scores.loc[scores.method.eq(reference_method), [
        "seed", "dataset", "site", "horizon", "forecast_kind", "mae", "rmse"
    ]].rename(columns={"mae": "reference_mae", "rmse": "reference_rmse"})
    gains = scores.merge(reference, on=["seed", "dataset", "site", "horizon", "forecast_kind"], how="left", validate="many_to_one")
    gains["mae_gain_percent"] = np.where(
        gains.reference_mae.eq(0), np.nan, 100.0 * (1.0 - gains.mae / gains.reference_mae)
    )
    gains["rmse_gain_percent"] = np.where(
        gains.reference_rmse.eq(0), np.nan, 100.0 * (1.0 - gains.rmse / gains.reference_rmse)
    )
    return scores, gains


def aggregate_gains(
    predictions: pd.DataFrame,
    *,
    methods: list[str],
    reference_method: str = "E1_original_chronos2",
) -> pd.DataFrame:
    """Return independently recomputed per-site, per-seed, and per-dataset gains."""
    require_shared_predictions(predictions, methods)
    if reference_method not in methods:
        raise ValueError("reference method must be included")
    selected = predictions.loc[predictions.method.isin(methods) & predictions.q.eq(0.5)].copy()
    lead_counts = selected.groupby(
        ["method", "seed", "dataset", "site", "origin"], dropna=False
    ).lead.agg(lambda values: set(values))
    if not lead_counts.map(lambda values: values == set(range(1, 25))).all():
        raise ValueError("every scored origin must contain exactly leads 1..24")

    rows: list[dict] = []
    levels = {
        "site": ["seed", "dataset", "site"],
        "seed": ["seed"],
        "dataset": ["dataset"],
    }
    for level, group_columns in levels.items():
        for horizon in (1, 4, 24):
            prefix = selected.loc[selected.lead.le(horizon)]
            for forecast_kind in ("raw", "post"):
                metric_rows = []
                for keys, group in prefix.groupby(["method", *group_columns], dropna=False, sort=True):
                    keys = keys if isinstance(keys, tuple) else (keys,)
                    metric_rows.append({
                        "method": keys[0],
                        **dict(zip(group_columns, keys[1:], strict=True)),
                        "mae": mae(group.y_true, group[f"forecast_{forecast_kind}"]),
                        "rmse": rmse(group.y_true, group[f"forecast_{forecast_kind}"]),
                    })
                metric_table = pd.DataFrame(metric_rows)
                reference = metric_table.loc[metric_table.method.eq(reference_method)].drop(
                    columns="method"
                ).rename(columns={"mae": "reference_mae", "rmse": "reference_rmse"})
                merged = metric_table.merge(reference, on=group_columns, how="left", validate="many_to_one")
                for metric in ("mae", "rmse"):
                    merged[f"{metric}_gain_percent"] = np.where(
                        merged[f"reference_{metric}"].eq(0),
                        np.nan,
                        100.0 * (1.0 - merged[metric] / merged[f"reference_{metric}"]),
                    )
                merged["aggregation"] = level
                merged["horizon"] = horizon
                merged["forecast_kind"] = forecast_kind
                rows.extend(merged.to_dict("records"))
    return pd.DataFrame(rows)
