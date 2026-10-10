"""Complete-array metrics using the accepted MMSP evaluation definition."""

import numpy as np


def metric(prediction, targets):
    prediction, targets = prediction.astype(np.float64), targets.astype(np.float64)
    error = prediction - targets
    variance = np.square(targets - targets.mean()).sum()
    return {
        'test_windows': len(targets),
        'mae': float(np.abs(error).mean()),
        'rmse': float(np.sqrt(np.square(error).mean())),
        'r2': float(1 - np.square(error).sum() / variance) if variance > 0 else None,
    }
