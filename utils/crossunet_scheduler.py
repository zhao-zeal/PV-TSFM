"""Match the optimizer state actually used by official lradj='type3'."""

from torch.optim.lr_scheduler import LambdaLR


def type3_factor(completed_epochs):
    # Constructing the unused OneCycleLR sets lr=max_lr/25 before epoch one.
    if completed_epochs == 0:
        return 1 / 25
    return 1.0 if completed_epochs < 3 else 0.9 ** (completed_epochs - 3)


class CrossUnetType3LR(LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, type3_factor, last_epoch=last_epoch)
