import torch
import torch.nn as nn

bce_logits = nn.BCEWithLogitsLoss()

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1,2,3))
    den = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps
    return 1.0 - (num / den).mean()

def combo_bce_dice(logits, targets, w_bce=0.5):
    return w_bce * bce_logits(logits, targets) + (1.0 - w_bce) * dice_loss_from_logits(logits, targets)
