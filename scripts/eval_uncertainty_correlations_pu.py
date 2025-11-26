#!/usr/bin/env python
"""
eval_uncertainty_correlations_pu.py

Compute disentanglement metrics for AU/EU/Soft on LIDC + PU vs GT:

Per slice:
    r(AU, soft), r(EU, soft), r(AU, EU)
    r(PU, GT)  # predictive uncertainty vs ground-truth uncertainty (disagreement)
    + log WHY correlation is NaN (too few voxels, var==0, etc.)

Per patient:
    mean over slices (nanmean) for all metrics

Dataset-level:
    mean ± std over patients

Outputs (in CKPT_DIR):
    - uncertainty_correlations_all.csv
    - uncertainty_correlations_valid.csv
    - uncertainty_correlations_nan.csv
    - uncertainty_correlations_patient.csv
"""

import os
import sys
import csv
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  # kept for potential future viz
from torch.utils.data import DataLoader, Subset

# ------------------------------------------------------
# HARD-CODED PATHS / SETTINGS
# ------------------------------------------------------
CKPT_DIR = "/home/boeke/Desktop/lidc_project/results/models"
DATA_ROOT = "/home/boeke/Desktop/lidc_project/data/LIDC-IDRI-slices"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = 16  # UNet base features
# ------------------------------------------------------

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.models.unet3d import UNet3D   # type: ignore
from data.dataloader import LIDCKaggleDataset  # type: ignore

# ---------------- CONFIG ----------------
TARGET_SHAPE = (16, 128, 128)
SPLIT_SEED = 666
SOFT_CLIP = 0.05          # ignore voxels with soft <= s or >= 1-s
MIN_SLICE_VOXELS = 10     # need at least this many valid voxels per slice
VIS_ROWS = 2              # reserved for future visualization
VIS_MAX = 5
# ----------------------------------------


@torch.no_grad()
def bernoulli_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Binary entropy H(p) = -p log p - (1-p) log (1-p)
    p: probabilities in [0,1], any shape
    """
    p = torch.clamp(p, eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def collate_fn(batch):
    """
    Custom collate for batch_size=1.

    Dataset item is a dict with:
      - "image": [D,H,W]
      - "target": [D,H,W]  (soft labels)
      - "disagreement": [D,H,W] (ground-truth uncertainty)
      - "meta": {"patient": ..., "nodule": ...}

    We:
      - add batch+channel dims
      - resample to TARGET_SHAPE
    """
    b = batch[0]
    img = b["image"].unsqueeze(0).unsqueeze(0)     # [1,1,D,H,W]
    tgt = b["target"].unsqueeze(0).unsqueeze(0)    # [1,1,D,H,W]
    dis = b["disagreement"].unsqueeze(0).unsqueeze(0)

    img = F.interpolate(img, size=TARGET_SHAPE, mode="trilinear", align_corners=False)
    tgt = F.interpolate(tgt, size=TARGET_SHAPE, mode="nearest")   # keep labels discrete/soft
    dis = F.interpolate(dis, size=TARGET_SHAPE, mode="nearest")

    return img, tgt, dis, b["meta"]


def build_val_loader(root: str) -> DataLoader:
    ds = LIDCKaggleDataset(root=root, agg="soft", return_all_masks=False)
    idx = np.arange(len(ds))
    np.random.seed(SPLIT_SEED)
    np.random.shuffle(idx)
    val = Subset(ds, idx[int(0.8 * len(ds)):])
    return DataLoader(val, batch_size=1, shuffle=False, collate_fn=collate_fn)


def load_models(ckpt_dir: str, base: int, device: str):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "unet3d_best_seed*.pth")))
    assert len(paths) > 0, f"No checkpoints found in {ckpt_dir}!"
    models = []
    for p in paths:
        m = UNet3D(base=base).to(device)
        state = torch.load(p, map_location=device)
        # support both {"state_dict": ...} and raw state_dict
        m.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def ensemble_forward(models, img: torch.Tensor) -> torch.Tensor:
    """
    models: list of UNet3D
    img: [1,1,D,H,W]
    returns: [K,1,D,H,W] probabilities
    """
    probs = [torch.sigmoid(m(img)) for m in models]  # if model output is logits
    return torch.stack(probs, dim=0)


def safe_slice(t: torch.Tensor) -> np.ndarray:
    """
    t: torch tensor [1,1,D,H,W] or [1,D,H,W]
    -> np.ndarray [D,H,W]
    """
    return t.squeeze().cpu().numpy()


def filtered_corr(x, y, soft, soft_clip: float, min_voxels: int):
    """
    Compute Pearson corr(x, y) on voxels where soft is in (soft_clip, 1-soft_clip).

    x, y, soft: array-like, same shape (broadcastable), will be flattened.

    Returns:
      (corr_value, reason)
      corr_value: float or np.nan
      reason: "" if OK, or a string like "too_few_valid_voxels", "var(x)==0", etc.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    soft = np.asarray(soft).ravel()

    mask = (soft > soft_clip) & (soft < 1.0 - soft_clip)
    if mask.sum() < min_voxels:
        return np.nan, "too_few_valid_voxels"

    x_valid = x[mask]
    y_valid = y[mask]

    if np.std(x_valid) == 0:
        return np.nan, "var(x)==0"
    if np.std(y_valid) == 0:
        return np.nan, "var(y)==0"

    r = np.corrcoef(x_valid, y_valid)[0, 1]
    if np.isnan(r):
        return np.nan, "nan_after_corrcoef"

    return float(r), ""


def corr_alea_epi(alea_slice, epi_slice):
    """
    Simple Pearson corr between AU and EU (no soft_clip).
    """
    a = np.asarray(alea_slice).ravel()
    e = np.asarray(epi_slice).ravel()
    if np.std(a) == 0 or np.std(e) == 0:
        return np.nan
    return float(np.corrcoef(a, e)[0, 1])


def corr_pu_gt(pu_slice, gt_slice, min_voxels: int = MIN_SLICE_VOXELS):
    """
    Pearson corr between predictive uncertainty (PU) and ground-truth uncertainty (disagreement).

    Here, PU is predictive entropy H(p_mean) and GT is disagreement.

    No soft_clip is applied; we just require:
      - at least min_voxels voxels
      - non-zero variance in both x and y
    """
    x = np.asarray(pu_slice).ravel()
    y = np.asarray(gt_slice).ravel()

    mask = ~np.isnan(y)
    if mask.sum() < min_voxels:
        return np.nan, "too_few_valid_voxels"

    x_valid = x[mask]
    y_valid = y[mask]

    if np.std(x_valid) == 0:
        return np.nan, "var(pu)==0"
    if np.std(y_valid) == 0:
        return np.nan, "var(gt)==0"

    r = np.corrcoef(x_valid, y_valid)[0, 1]
    if np.isnan(r):
        return np.nan, "nan_after_corrcoef"

    return float(r), ""


def compute_slice_corrs(alea_slice, epi_slice, soft_slice, pu_slice, gt_slice):
    """
    alea_slice, epi_slice, soft_slice, pu_slice, gt_slice: np.ndarray [H,W] for a single slice.
    """
    r_alea_soft, why_alea = filtered_corr(
        alea_slice,
        soft_slice,
        soft_slice,
        soft_clip=SOFT_CLIP,
        min_voxels=MIN_SLICE_VOXELS,
    )
    r_epi_soft, why_epi = filtered_corr(
        epi_slice,
        soft_slice,
        soft_slice,
        soft_clip=SOFT_CLIP,
        min_voxels=MIN_SLICE_VOXELS,
    )
    r_alea_epi = corr_alea_epi(alea_slice, epi_slice)

    r_pu_gt, why_pu = corr_pu_gt(
        pu_slice,
        gt_slice,
        min_voxels=MIN_SLICE_VOXELS,
    )

    return {
        "r_alea_soft": r_alea_soft,
        "r_epi_soft": r_epi_soft,
        "r_alea_epi": r_alea_epi,
        "r_pu_gt": r_pu_gt,
        "r_alea_soft_why": why_alea,
        "r_epi_soft_why": why_epi,
        "r_pu_gt_why": why_pu,
    }


def main():
    device = DEVICE
    ckpt_dir = CKPT_DIR
    data_root = DATA_ROOT
    base = BASE

    models = load_models(ckpt_dir, base, device)
    loader = build_val_loader(data_root)

    rows_all = []
    patient_level_stats = []  # (patient, nodule, mean_alea, mean_epi, mean_pu_gt)

    print("Running evaluation…\n")

    for img, tgt, dis, meta in loader:
        img = img.to(device)
        tgt = tgt.to(device)
        # disagreement 'dis' used only after safe_slice, no need to move to device

        patient = meta["patient"]
        nodule = meta["nodule"]

        # ----- Forward pass -----
        ps = ensemble_forward(models, img)              # [K,1,D,H,W]
        p_mean = ps.mean(0)                             # [1,1,D,H,W]
        alea = bernoulli_entropy(ps).mean(0)            # E[H[p]]
        epi = bernoulli_entropy(p_mean) - alea          # mutual information

        # Predictive uncertainty = entropy of ensemble mean probability
        pu = bernoulli_entropy(p_mean)                  # [1,1,D,H,W]

        soft_np = safe_slice(tgt)       # [D,H,W]
        alea_np = safe_slice(alea)      # [D,H,W]
        epi_np = safe_slice(epi)        # [D,H,W]
        pu_np = safe_slice(pu)          # [D,H,W]
        gt_np = safe_slice(dis)         # [D,H,W]  (disagreement)

        D = soft_np.shape[0]
        r_alea_slices = []
        r_epi_slices = []
        r_pu_gt_slices = []

        for d in range(D):
            a_slice = alea_np[d]   # [H,W]
            e_slice = epi_np[d]    # [H,W]
            s_slice = soft_np[d]   # [H,W]
            pu_slice = pu_np[d]    # [H,W]
            gt_slice = gt_np[d]    # [H,W]

            res = compute_slice_corrs(a_slice, e_slice, s_slice, pu_slice, gt_slice)

            row = {
                "patient": patient,
                "nodule": nodule,
                "slice": int(d),
                "r_alea_soft": res["r_alea_soft"],
                "r_epi_soft": res["r_epi_soft"],
                "r_alea_epi": res["r_alea_epi"],
                "r_pu_gt": res["r_pu_gt"],
                "r_alea_soft_why": res["r_alea_soft_why"],
                "r_epi_soft_why": res["r_epi_soft_why"],
                "r_pu_gt_why": res["r_pu_gt_why"],
            }

            rows_all.append(row)

            r_alea_slices.append(res["r_alea_soft"])
            r_epi_slices.append(res["r_epi_soft"])
            r_pu_gt_slices.append(res["r_pu_gt"])

        mean_alea = np.nanmean(r_alea_slices) if len(r_alea_slices) else np.nan
        mean_epi = np.nanmean(r_epi_slices) if len(r_epi_slices) else np.nan
        mean_pu_gt = np.nanmean(r_pu_gt_slices) if len(r_pu_gt_slices) else np.nan

        patient_level_stats.append((patient, nodule, mean_alea, mean_epi, mean_pu_gt))

        print(
            f"{patient}/{nodule}  →  "
            f"AU-soft={mean_alea:.3f}, "
            f"EU-soft={mean_epi:.3f}, "
            f"PU-GT={mean_pu_gt:.3f}"
        )

    # ----- Save all-slice CSV -----
    out_all = os.path.join(ckpt_dir, "uncertainty_correlations_all.csv")

    if len(rows_all) == 0:
        raise RuntimeError("No rows collected – did the loader return anything?")

    all_keys = set()
    for r in rows_all:
        all_keys.update(r.keys())
    keys = sorted(all_keys)

    with open(out_all, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_all:
            w.writerow(r)

    print("\nSaved:", out_all)

    # ------------------------------------------------
    #   CREATE: valid.csv   and   nan.csv
    # ------------------------------------------------
    valid_rows = []
    nan_rows = []

    for r in rows_all:
        r_alea = r["r_alea_soft"]
        r_epi = r["r_epi_soft"]
        # validity defined exactly as in original script
        if (not np.isnan(r_alea)) and (not np.isnan(r_epi)):
            valid_rows.append(r)
        else:
            nan_rows.append(r)

    out_valid = os.path.join(ckpt_dir, "uncertainty_correlations_valid.csv")
    with open(out_valid, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in valid_rows:
            w.writerow(r)

    out_nan = os.path.join(ckpt_dir, "uncertainty_correlations_nan.csv")
    with open(out_nan, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in nan_rows:
            w.writerow(r)

    print("Saved:", out_valid)
    print("Saved:", out_nan)

    # ------------------------------------------------
    #   CREATE: patient-level summary CSV
    # ------------------------------------------------
    patient_summary = {}
    for r in rows_all:
        pid = r["patient"]
        nid = r["nodule"]
        key = (pid, nid)

        if key not in patient_summary:
            patient_summary[key] = {
                "patient": pid,
                "nodule": nid,
                "alea_list": [],
                "epi_list": [],
                "pu_gt_list": [],
                "alea_epi_list": [],
                "total_slices": 0,
                "valid_slices": 0,
            }

        dct = patient_summary[key]
        dct["total_slices"] += 1

        rA = r["r_alea_soft"]
        rE = r["r_epi_soft"]
        rAE = r["r_alea_epi"]
        rPU = r.get("r_pu_gt", np.nan)

        if not np.isnan(rA):
            dct["alea_list"].append(rA)
        if not np.isnan(rE):
            dct["epi_list"].append(rE)
        if not np.isnan(rAE):
            dct["alea_epi_list"].append(rAE)
        if not np.isnan(rPU):
            dct["pu_gt_list"].append(rPU)

        if (not np.isnan(rA)) and (not np.isnan(rE)):
            dct["valid_slices"] += 1

    patient_rows = []
    for key, d in patient_summary.items():
        alea_mean = np.nanmean(d["alea_list"]) if len(d["alea_list"]) else np.nan
        epi_mean = np.nanmean(d["epi_list"]) if len(d["epi_list"]) else np.nan
        alea_epi_mean = (
            np.nanmean(d["alea_epi_list"]) if len(d["alea_epi_list"]) else np.nan
        )
        pu_gt_mean = (
            np.nanmean(d["pu_gt_list"]) if len(d["pu_gt_list"]) else np.nan
        )

        patient_rows.append(
            {
                "patient": d["patient"],
                "nodule": d["nodule"],
                "mean_r_alea_soft": alea_mean,
                "mean_r_epi_soft": epi_mean,
                "mean_r_alea_epi": alea_epi_mean,
                "mean_r_pu_gt": pu_gt_mean,
                "total_slices": d["total_slices"],
                "valid_slices": d["valid_slices"],
                "valid_ratio": d["valid_slices"] / d["total_slices"]
                if d["total_slices"] > 0
                else np.nan,
                "all_nan": d["valid_slices"] == 0,
            }
        )

    if len(patient_rows) == 0:
        raise RuntimeError("No patient-level rows – something is wrong with rows_all.")

    patient_keys = list(patient_rows[0].keys())
    out_patient = os.path.join(ckpt_dir, "uncertainty_correlations_patient.csv")
    with open(out_patient, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=patient_keys)
        w.writeheader()
        for r in patient_rows:
            w.writerow(r)

    print("Saved:", out_patient)

    # ------------------------------------------------
    #   Dataset-level summary (from patient_rows)
    # ------------------------------------------------
    print("\n=== SUMMARY (patient level) ===")
    alea_vals = np.array([r["mean_r_alea_soft"] for r in patient_rows], dtype=float)
    epi_vals = np.array([r["mean_r_epi_soft"] for r in patient_rows], dtype=float)
    pu_gt_vals = np.array([r["mean_r_pu_gt"] for r in patient_rows], dtype=float)

    print(
        f"Aleatoric-soft: {np.nanmean(alea_vals):.4f} ± {np.nanstd(alea_vals):.4f}"
    )
    print(
        f"Epistemic-soft: {np.nanmean(epi_vals):.4f} ± {np.nanstd(epi_vals):.4f}"
    )
    print(
        f"PU-GT:          {np.nanmean(pu_gt_vals):.4f} ± {np.nanstd(pu_gt_vals):.4f}"
    )


if __name__ == "__main__":
    main()
