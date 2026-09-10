"""Protocol-facing data types with no model-side metadata features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class IntervalSemantics(str, Enum):
    AVERAGE_POWER = "average_power"
    INTERVAL_ENERGY = "interval_energy"


class TimestampLabel(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class PVSeries:
    """One physical system represented only by timestamps, target and observation mask."""

    dataset_id: str
    physical_site_id: str
    timestamp: pd.DatetimeIndex
    target: np.ndarray
    observed_mask: np.ndarray
    unit: str | None
    timezone: str | None
    interval_semantics: IntervalSemantics | None
    timestamp_label: TimestampLabel | None
    native_interval_minutes: int
    physical_location_group_id: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        timestamp = pd.DatetimeIndex(self.timestamp)
        target = np.asarray(self.target, dtype=np.float64)
        mask = np.asarray(self.observed_mask, dtype=bool)
        if timestamp.tz is not None and str(timestamp.tz) != "UTC":
            raise ValueError("canonical PVSeries timestamps must be UTC")
        if len(timestamp) != len(target) or len(target) != len(mask):
            raise ValueError("timestamp, target and observed_mask lengths differ")
        if timestamp.has_duplicates or not timestamp.is_monotonic_increasing:
            raise ValueError("timestamps must be unique and increasing")
        if self.native_interval_minutes <= 0:
            raise ValueError("native_interval_minutes must be positive")
        if np.any(mask & ~np.isfinite(target)):
            raise ValueError("observed target values must be finite")
        if np.any(~mask & np.isfinite(target)):
            raise ValueError("unobserved target values must be NaN")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "observed_mask", mask)

    def model_inputs(self) -> dict[str, np.ndarray]:
        """Return the complete feature allowlist and nothing else."""
        return {
            "past_target": self.target.copy(),
            "past_observed_mask": self.observed_mask.copy(),
        }
