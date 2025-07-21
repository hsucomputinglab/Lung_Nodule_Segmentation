import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import compute_dice, compute_iou


def iou_score(pred, target):
    smooth = 1e-5

    if torch.is_tensor(pred):
        pred = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    pred_ = pred > 0.5
    target_ = target
    intersection = (pred_ * target_).sum()
    union = pred_.sum() + target_.sum() - intersection

    return (intersection + smooth) / (union + smooth)
    # iou_coef = compute_iou(pred, target, ignore_empty=False).mean(dim=0)
    # return iou_coef.item()

# Dice
def dice_score_train(pred, target):
    # we need to use sigmoid because the output of Unet is logit.
    # pred = torch.sigmoid(pred).view(-1).data.cpu().numpy()
    # target = target.view(-1).data.cpu().numpy()
    # pred = pred > 0.5
    # dice_coef = compute_dice(pred, target, ignore_empty=False).mean()
    # return dice_coef.item()
    smooth = 1e-5

    pred = torch.sigmoid(pred).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (pred * target).sum()

    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def dice_score_validation(pred, target):
    # pred = pred.view(-1)
    # pred = (pred>0.5).float().cpu().numpy()
    # target = target.view(-1).data.cpu().numpy()
    # dice_coef = compute_dice(pred, target, ignore_empty=False).mean()
    # return dice_coef.item()
    smooth = 1e-5

    pred = pred.view(-1)
    pred = (pred > 0.5).float().cpu().numpy()
    # pred = torch.sigmoid(pred).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (pred * target).sum()

    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Recall
def recall(pred, target, smooth=1e-5):
    """
    Computes the recall (sensitivity) between predicted and target masks.

    Args:
        pred (torch.Tensor): Predicted mask (values between 0 and 1).
        target (torch.Tensor): Ground truth mask (binary, 0 or 1).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Recall score.
    """
    if torch.is_tensor(pred):
        pred = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    pred = pred > 0.5  # Binarize the prediction at threshold 0.5
    target = target

    true_positive = (pred * target).sum()
    total_actual_positive = target.sum()

    recall = (true_positive + smooth) / (total_actual_positive + smooth)
    return recall.mean()  # Average recall over the batch


# Precision
def precision(pred, target, smooth=1e-5):
    """
    Computes the precision between predicted and target masks.

    Args:
        pred (torch.Tensor): Predicted mask (values between 0 and 1).
        target (torch.Tensor): Ground truth mask (binary, 0 or 1).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Precision score.
    """
    if torch.is_tensor(pred):
        pred = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    pred = pred > 0.5  # Binarize the prediction at threshold 0.5
    target = target

    true_positive = (pred * target).sum()
    total_pred_positive = pred.sum()

    precision = (true_positive + smooth) / (total_pred_positive + smooth)
    return precision.mean()  # Average precision over the batch


# F1-Score
def f1_score(pred, target, smooth=1e-5):
    """
    Computes the F1-Score (harmonic mean of precision and recall).

    Args:
        pred (torch.Tensor): Predicted mask (values between 0 and 1).
        target (torch.Tensor): Ground truth mask (binary, 0 or 1).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: F1-Score.
    """
    if torch.is_tensor(pred):
        pred = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    prec = precision(pred, target, smooth)
    rec = recall(pred, target, smooth)

    f1 = 2 * (prec * rec) / (prec + rec)
    return f1
