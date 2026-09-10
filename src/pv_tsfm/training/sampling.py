"""Deterministic hierarchical source/location/system/window sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = {"original_source_id", "physical_location_id", "physical_site_id", "window_id"}


def build_window_sampling_manifest(
    eligible_windows: pd.DataFrame,
    *,
    seed: int,
    optimizer_steps: int = 1000,
    effective_batch: int = 128,
) -> pd.DataFrame:
    missing = REQUIRED - set(eligible_windows.columns)
    if missing:
        raise ValueError(f"eligible windows missing sampling columns: {sorted(missing)}")
    if eligible_windows.empty or eligible_windows.window_id.duplicated().any():
        raise ValueError("eligible window IDs must be non-empty and unique")
    rng = np.random.default_rng(seed)
    grouped = {}
    for source, source_rows in eligible_windows.groupby("original_source_id", sort=True):
        grouped[source] = {}
        for location, location_rows in source_rows.groupby("physical_location_id", sort=True):
            grouped[source][location] = {
                site: rows.window_id.to_numpy()
                for site, rows in location_rows.groupby("physical_site_id", sort=True)
            }
    sources = sorted(grouped)
    rows = []
    for step in range(1, optimizer_steps + 1):
        for batch_position in range(effective_batch):
            source = sources[rng.integers(len(sources))]
            locations = sorted(grouped[source])
            location = locations[rng.integers(len(locations))]
            sites = sorted(grouped[source][location])
            site = sites[rng.integers(len(sites))]
            windows = grouped[source][location][site]
            window_id = windows[rng.integers(len(windows))]
            rows.append({
                "seed": seed,
                "optimizer_step": step,
                "batch_position": batch_position,
                "original_source_id": source,
                "physical_location_id": location,
                "physical_site_id": site,
                "window_id": window_id,
            })
    return pd.DataFrame(rows)
