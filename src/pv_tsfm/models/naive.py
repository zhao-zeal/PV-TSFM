"""Frozen E0 power-only baselines."""

from __future__ import annotations

import numpy as np


def _validate(past_target: np.ndarray, past_observed_mask: np.ndarray, horizon: int) -> np.ndarray:
    target = np.asarray(past_target, dtype=np.float64)
    mask = np.asarray(past_observed_mask, dtype=bool)
    if target.ndim != 1 or mask.shape != target.shape:
        raise ValueError("past target and mask must be aligned one-dimensional arrays")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not mask.all() or not np.isfinite(target).all():
        raise ValueError("MVP main-track baselines require complete context")
    return target


class LastValue:
    method = "E0_last_value"

    def predict(self, past_target: np.ndarray, past_observed_mask: np.ndarray, horizon: int) -> np.ndarray:
        target = _validate(past_target, past_observed_mask, horizon)
        return np.repeat(target[-1], horizon)


class DailySeasonalPersistence:
    method = "E0_daily_seasonal_persistence"

    def predict(self, past_target: np.ndarray, past_observed_mask: np.ndarray, horizon: int) -> np.ndarray:
        target = _validate(past_target, past_observed_mask, horizon)
        if len(target) < 24:
            raise ValueError("daily persistence requires at least 24 context points")
        return np.resize(target[-24:], horizon)


class SevenDaySameHourMean:
    method = "E0_past_7_days_same_hour_mean"

    def predict(self, past_target: np.ndarray, past_observed_mask: np.ndarray, horizon: int) -> np.ndarray:
        target = _validate(past_target, past_observed_mask, horizon)
        if len(target) < 7 * 24:
            raise ValueError("seven-day same-hour mean requires at least 168 context points")
        profile = target[-7 * 24 :].reshape(7, 24).mean(axis=0)
        return np.resize(profile, horizon)
