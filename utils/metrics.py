"""Complete-array metrics using the accepted MMSP evaluation definition."""

import numpy as np


def metric(prediction, targets):
    if prediction.shape != targets.shape or targets.size == 0:
        raise ValueError(f'Prediction/target shapes must match and be nonempty: {prediction.shape}, {targets.shape}')
    if not np.isfinite(prediction).all() or not np.isfinite(targets).all():
        raise ValueError('Predictions and targets must be finite')
    prediction, targets = prediction.astype(np.float64), targets.astype(np.float64)
    error = prediction - targets
    variance = np.square(targets - targets.mean()).sum()
    result = {
        'test_windows': len(targets),
        'mae': float(np.abs(error).mean()),
        'rmse': float(np.sqrt(np.square(error).mean())),
        'r2': float(1 - np.square(error).sum() / variance) if variance > 0 else None,
    }
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise ValueError('Metric calculation produced nonfinite values')
    return result
