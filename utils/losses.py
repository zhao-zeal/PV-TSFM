"""Original regression criteria, including FusionSF's training VQ auxiliary loss."""

from torch.nn import functional as F


def forecasting_loss(result, targets, model_name):
    prediction = result['prediction'] if isinstance(result, dict) else result
    if model_name == 'FusionSF':
        return F.l1_loss(prediction, targets) + result['vq_loss']
    return F.mse_loss(prediction, targets)
