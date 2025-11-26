#!/usr/bin/env python

"""
viz_all_slices_grid_doubleslice_cbars.py

Visualizes all slices for each validation nodule:

Each row contains TWO slices:
    [CT + ann, Soft, AU, EU]  |  [CT + ann, Soft, AU, EU]

=> 8 panels per row, color-coded with fixed scales and shared colorbars.

Output:
  results/models/uncertainty_doubleslice_cbars_case<k>_<patient>_<nodule>.png
"""

import os, sys, glob, math, numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- basic config ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR     = os.path.join(PROJECT_ROOT, "results", "models")
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data", "LIDC-IDRI-slices")

TARGET_SHAPE = (16, 128, 128)
SPLIT_SEED   = 666
N_PATIENTS   = 10
SLICES_PER_ROW = 2

SOFT_VMIN, SOFT_VMAX = 0.0, 1.0
AU_VMIN,   AU_VMAX   = 0.0, 0.7
EU_VMIN,   EU_VMAX   = 0.0, 0.03
CMAP = "magma"

sys.path.append(PROJECT_ROOT)
from scripts.models.unet3d import UNet3D
from data.dataloader import LIDCKaggleDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Using device:", device)


@torch.no_grad()
def bernoulli_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


@torch.no_grad()
def ensemble_forward(models, img: torch.Tensor) -> torch.Tensor:
    probs = [torch.sigmoid(m(img)) for m in models]
    return torch.stack(probs, dim=0)


def resize_3d(x: torch.Tensor, target_shape):
    x = x.unsqueeze(0).unsqueeze(0).float()
    x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
    return x.squeeze(0).squeeze(0)


def resize_masks(masks: torch.Tensor, target_shape):
    masks = masks.unsqueeze(1).float()
    masks = F.interpolate(masks, size=target_shape, mode="trilinear", align_corners=False)
    return masks.squeeze(1)


# ---------------- load ensemble ----------------
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "unet3d_best_seed*.pth")))
if not ckpts:
    raise RuntimeError(f"No checkpoints in {CKPT_DIR}")
print(f"[INFO] Found {len(ckpts)} ensemble members:")
for p in ckpts:
    print("  -", p)

models = []
for p in ckpts:
    m = UNet3D(base=16).to(device)
    state = torch.load(p, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        m.load_state_dict(state["state_dict"])
    else:
        m.load_state_dict(state)
    m.eval()
    models.append(m)

# ---------------- load dataset ----------------
ds = LIDCKaggleDataset(root=DATA_ROOT, agg="soft", return_all_masks=True, normalize=True)
n_total = len(ds)
n_train = int(0.8 * n_total)
idx_all = np.arange(n_total)
np.random.seed(SPLIT_SEED)
np.random.shuffle(idx_all)
val_indices = idx_all[n_train:]
print(f"[INFO] Dataset size={n_total}, val size={len(val_indices)}")

colors_contours = ["cyan", "magenta", "lime", "yellow", "red", "blue"]

# ---------------- main loop ----------------
for ex_idx in range(min(N_PATIENTS, len(val_indices))):
    raw_idx = val_indices[ex_idx]
    sample = ds[raw_idx]

    img_vol   = sample["image"]
    soft_vol  = sample["target"]
    masks_vol = sample["masks"]
    meta      = sample["meta"]

    patient, nodule = meta["patient"], meta["nodule"]
    print(f"\n[INFO] Case {ex_idx+1}/{N_PATIENTS}: {patient}/{nodule} (raw idx {raw_idx})")

    img_res   = resize_3d(img_vol, TARGET_SHAPE)
    soft_res  = resize_3d(soft_vol, TARGET_SHAPE)
    masks_res = resize_masks(masks_vol, TARGET_SHAPE)

    img_t = img_res.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs  = ensemble_forward(models, img_t)   # [K,1,D,H,W]
        p_mean = probs.mean(0)
        alea   = bernoulli_entropy(probs).mean(0)
        p_ent  = bernoulli_entropy(p_mean)
        epi    = p_ent - alea

    img_np   = img_res.cpu().numpy()
    soft_np  = soft_res.cpu().numpy()
    alea_np  = alea.squeeze().cpu().numpy()
    epi_np   = epi.squeeze().cpu().numpy()
    masks_np = masks_res.cpu().numpy()

    D = img_np.shape[0]
    rows = math.ceil(D / SLICES_PER_ROW)
    cols = 4 * SLICES_PER_ROW

    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 1.6 * rows))
    if rows == 1:
        axes = axes.reshape(1, cols)

    im_soft0 = im_au0 = im_eu0 = None

    for z in range(D):
        r = z // SLICES_PER_ROW
        s = z % SLICES_PER_ROW
        base_col = s * 4

        # CT
        ax_ct = axes[r, base_col]
        ax_ct.imshow(img_np[z], cmap="gray", origin="lower")
        for a in range(masks_np.shape[0]):
            ax_ct.contour(masks_np[a, z], levels=[0.5],
                          colors=[colors_contours[a % len(colors_contours)]],
                          linewidths=0.8)
        ax_ct.set_ylabel(f"z={z}", rotation=0, labelpad=12, fontsize=8)
        ax_ct.axis("off")

        # Soft
        ax_s = axes[r, base_col + 1]
        im_soft = ax_s.imshow(soft_np[z], origin="lower",
                              vmin=SOFT_VMIN, vmax=SOFT_VMAX, cmap=CMAP)
        ax_s.axis("off")
        if im_soft0 is None:
            im_soft0 = im_soft

        # Aleatoric
        ax_au = axes[r, base_col + 2]
        im_au = ax_au.imshow(alea_np[z], origin="lower",
                             vmin=AU_VMIN, vmax=AU_VMAX, cmap=CMAP)
        ax_au.axis("off")
        if im_au0 is None:
            im_au0 = im_au

        # Epistemic
        ax_eu = axes[r, base_col + 3]
        im_eu = ax_eu.imshow(epi_np[z], origin="lower",
                             vmin=EU_VMIN, vmax=EU_VMAX, cmap=CMAP)
        ax_eu.axis("off")
        if im_eu0 is None:
            im_eu0 = im_eu

    # titles
    axes[0, 0].set_title("CT + ann")
    axes[0, 1].set_title("Soft (0–1)")
    axes[0, 2].set_title(f"Aleatoric (0–{AU_VMAX})")
    axes[0, 3].set_title(f"Epistemic (0–{EU_VMAX})")

    fig.suptitle(f"{patient} / {nodule} – all slices (D={D}), 2 per row, cmap={CMAP}",
                 fontsize=11, y=0.99)

    # shared colorbars
    soft_axes = axes[:, 1::4].ravel().tolist()
    au_axes   = axes[:, 2::4].ravel().tolist()
    eu_axes   = axes[:, 3::4].ravel().tolist()
    fig.colorbar(im_soft0, ax=soft_axes, fraction=0.015, pad=0.01)
    fig.colorbar(im_au0,   ax=au_axes,   fraction=0.015, pad=0.02)
    fig.colorbar(im_eu0,   ax=eu_axes,   fraction=0.015, pad=0.03)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03,
                        wspace=0.02, hspace=0.08)

    out_name = f"uncertainty_doubleslice_cbars_case{ex_idx+1}_{patient}_{nodule}.png".replace("/", "_")
    out_path = os.path.join(CKPT_DIR, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")

print("\n[INFO] Done.")
