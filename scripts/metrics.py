# scripts/metrics.py
import torch
import torch.nn as nn

def soft_dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1,2,3,4))
    den = prob.pow(2).sum(dim=(1,2,3,4)) + target.pow(2).sum(dim=(1,2,3,4)) + eps
    return (1 - num / den).mean()

@torch.no_grad()
def dice_score(prob, target, thr=0.5, eps=1e-6):
    pred = (prob > thr).float()
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return (num / den).item()

@torch.no_grad()
def iou_score(prob, target, thr=0.5, eps=1e-6):
    pred = (prob > thr).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter + eps
    return (inter / union).item()

def make_pos_weight(target_ratio: float, min_weight: float = 1.0):
    p = max(min(target_ratio, 0.9999), 1e-4)
    w = (1.0 - p) / p
    return max(w, min_weight)

def bce_with_logits(pos_weight_value: float):
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value]))
