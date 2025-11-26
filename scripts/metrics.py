# scripts/metrics.py
import torch
import torch.nn as nn

def soft_dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3, 4))
    den = prob.pow(2).sum(dim=(1, 2, 3, 4)) + target.pow(2).sum(dim=(1, 2, 3, 4)) + eps
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


# -------------------------------------------------------------------------
#  Correlation between uncertainty maps and soft labels (with filtering)
# -------------------------------------------------------------------------

@torch.no_grad()
def corr_uncert_softlabel(
    uncert: torch.Tensor,
    soft_label: torch.Tensor,
    soft_clip: float = 0.05,
    min_voxels: int = 10,
    eps: float = 1e-6,
) -> float:
    
    if uncert.shape != soft_label.shape:
        raise ValueError(f"Shape mismatch: uncert {uncert.shape}, soft_label {soft_label.shape}")

    uncert = uncert.float()
    soft_label = soft_label.float()

    B = uncert.shape[0]
    u_flat = uncert.view(B, -1)
    s_flat = soft_label.view(B, -1)

    corrs = []
    for b in range(B):
        u = u_flat[b]
        s = s_flat[b]

        if soft_clip is not None:
            mask = (s > soft_clip) & (s < 1.0 - soft_clip)
        else:
            mask = torch.ones_like(s, dtype=torch.bool)

        if mask.sum() < min_voxels:
            continue

        u = u[mask]
        s = s[mask]

        # Center
        u_c = u - u.mean()
        s_c = s - s.mean()

        num = (u_c * s_c).sum()
        den = torch.sqrt(u_c.pow(2).sum() * s_c.pow(2).sum() + eps)

        if den <= eps:
            continue

        corrs.append(num / den)

    if len(corrs) == 0:
        return float("nan")

    return torch.stack(corrs).mean().item()


@torch.no_grad()
def corr_alea_softlabel(
    alea_hat: torch.Tensor,
    soft_label: torch.Tensor,
    soft_clip: float = 0.05,
    min_voxels: int = 10,
    eps: float = 1e-6,
) -> float:
    
    return corr_uncert_softlabel(alea_hat, soft_label, soft_clip, min_voxels, eps)


@torch.no_grad()
def corr_epi_softlabel(
    epi_hat: torch.Tensor,
    soft_label: torch.Tensor,
    soft_clip: float = 0.05,
    min_voxels: int = 10,
    eps: float = 1e-6,
) -> float:
    return corr_uncert_softlabel(epi_hat, soft_label, soft_clip, min_voxels, eps)
