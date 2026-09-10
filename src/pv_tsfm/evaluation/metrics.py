"""Basic point metrics in original target units."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    truth, prediction = _aligned(y_true, y_pred)
    return float(np.mean(np.abs(prediction - truth)))


def rmse(y_true, y_pred) -> float:
    truth, prediction = _aligned(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def _aligned(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("metric arrays must have the same non-empty shape")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("basic metrics require finite truth and predictions")
    return truth, prediction
