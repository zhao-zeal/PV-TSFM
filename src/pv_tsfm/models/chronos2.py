"""Official Chronos-2 E1 inference without any external normalization."""

from __future__ import annotations

from typing import Any

import numpy as np


MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


class Chronos2Forecaster:
    method = "E1_original_chronos2"

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        quantiles = list(self.pipeline.quantiles)
        if 0.5 not in quantiles:
            raise ValueError("locked Chronos-2 quantile grid has no median")

    @classmethod
    def from_pretrained(
        cls,
        *,
        device_map: str | dict = "cpu",
        torch_dtype: str = "auto",
        local_files_only: bool = True,
    ) -> "Chronos2Forecaster":
        from chronos import Chronos2Pipeline

        model_source = MODEL_ID
        if local_files_only:
            from huggingface_hub import snapshot_download

            model_source = snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=True,
            )
        pipeline = Chronos2Pipeline.from_pretrained(
            model_source,
            revision=None if local_files_only else MODEL_REVISION,
            device_map=device_map,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        return cls(pipeline)

    def predict(self, past_target: np.ndarray, past_observed_mask: np.ndarray, horizon: int) -> np.ndarray:
        target = np.asarray(past_target, dtype=np.float32)
        mask = np.asarray(past_observed_mask, dtype=bool)
        if target.ndim != 1 or mask.shape != target.shape:
            raise ValueError("past target and mask must be aligned one-dimensional arrays")
        if len(target) != 336:
            raise ValueError("primary Chronos-2 track requires exactly 336 context points")
        if horizon != 24:
            raise ValueError("primary Chronos-2 track requires horizon 24")
        context = np.where(mask, target, np.nan)
        quantile_forecasts, medians = self.pipeline.predict_quantiles(
            [context],
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=1,
            context_length=336,
            cross_learning=False,
        )
        if len(quantile_forecasts) != 1 or len(medians) != 1:
            raise RuntimeError("Chronos-2 returned an unexpected number of forecasts")
        median = np.asarray(medians[0], dtype=np.float64).squeeze()
        if median.shape != (horizon,) or not np.isfinite(median).all():
            raise RuntimeError(f"invalid Chronos-2 median shape/values: {median.shape}")
        return median
