"""Prediction and scoring utilities."""

from .prediction import predict_manifest_rows
from .scoring import aggregate_gains, compute_basic_scores, require_shared_predictions

__all__ = ["aggregate_gains", "compute_basic_scores", "predict_manifest_rows", "require_shared_predictions"]
