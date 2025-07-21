import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss as MonaiDiceLoss


# BCE Loss (Binary Cross-Entropy)
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # Apply BCE loss
        return self.bce(pred, target)


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.dice = MonaiDiceLoss(sigmoid=True)

    def forward(self, pred, target, smooth=1e-5):
        return self.dice(pred, target)



# Combined BCE + Dice Loss
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        # Apply BCE loss
        bce = self.bce_loss(pred, target)

        # Apply Dice loss
        dice = self.dice_loss(pred, target)

        # Combine BCE and Dice losses
        loss = bce + dice
        # loss = bce

        return loss, bce, dice
