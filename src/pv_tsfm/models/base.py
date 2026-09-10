"""Minimal prediction contract that cannot accept future truth or metadata features."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Forecaster(Protocol):
    method: str

    def predict(
        self,
        past_target: np.ndarray,
        past_observed_mask: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Return a point forecast using historical target and mask only."""
