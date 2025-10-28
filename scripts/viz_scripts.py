# scripts/viz_predictions.py
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from skimage import measure
import imageio.v2 as imageio  # for GIF creation
from dataloader import LIDCKaggleDataset


# ---- configuration (must match training) ----
TARGET_SHAPE = (16, 128, 128)     # (D, H, W)
AGG = "soft"                      # label aggregation: "soft" or "vote2"
CKPT_PATH = "results/models/unet3d_best.pth"

# ---- visualization settings ----
GREEN_IS_PRED = True              # green = prediction contours, red = GT
GIF_DURATION  = 0.35              # slower animation
SLICE_GT_THR  = 0.5               # threshold for binarizing GT
NUM_SAMPLES   = 8
THR           = 0.5
MAKE_GIF      = True


# ---------------- 3D U-Net ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = DoubleConv(1, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool = nn.MaxPool3d(2)
        self.bott = DoubleConv(base*4, base*8)
        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.out = nn.Conv3d(base, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bott(self.pool(e3))
        d3 = self.up3(b); d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)

# ---------------- utils ----------------
@torch.no_grad()
def dice_score(prob, target, thr=0.5, eps=1e-6):
    pred = (prob > thr).float()
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return (num / den).item()

def collate_resize(batch):
    b = batch[0]
    img = b["image"].unsqueeze(0).unsqueeze(0)
    tgt = b["target"].unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=TARGET_SHAPE, mode="trilinear", align_corners=False)
    tgt = F.interpolate(tgt, size=TARGET_SHAPE, mode="trilinear", align_corners=False)
    meta = b.get("meta", {})
    return img, tgt, meta

def make_preview(img_5d, tgt_5d, prob_5d, out_png, thr=0.5, title=""):
    img = img_5d.detach().cpu().numpy()[0,0]
    tgt = tgt_5d.detach().cpu().numpy()[0,0]
    prob = prob_5d.detach().cpu().numpy()[0,0]

    # pick a slice with most GT (fallback to pred if GT empty)
    vol_gt   = tgt.sum(axis=(1,2))
    vol_pred = (prob >= thr).sum(axis=(1,2))
    z = int(np.argmax(vol_gt if vol_gt.sum() > 0 else vol_pred))

    pred_bin = (prob[z] >= thr).astype(np.uint8)
    gt_bin   = (tgt[z]  >= SLICE_GT_THR).astype(np.uint8)

    plt.figure(figsize=(14,3))

    # 1) CT
    ax = plt.subplot(1,4,1)
    ax.imshow(img[z], cmap="gray"); ax.set_title("CT"); ax.axis("off")

    # 2) GT overlay (show soft GT so disagreement is visible)
    ax = plt.subplot(1,4,2)
    ax.imshow(img[z], cmap="gray")
    ax.imshow(tgt[z], cmap="Greens", alpha=0.35, vmin=0, vmax=1)
    ax.set_title("GT overlay"); ax.axis("off")

    # 3) Pred prob
    ax = plt.subplot(1,4,3)
    ax.imshow(prob[z]); ax.set_title("Pred prob"); ax.axis("off")

    # 4) Contours (swap colors if GREEN_IS_PRED)
    ax = plt.subplot(1,4,4)
    ax.imshow(img[z], cmap="gray")
    if GREEN_IS_PRED:
        pred_color, gt_color = "lime", "red"
        pred_label, gt_label = "Pred", "GT"
    else:
        pred_color, gt_color = "red", "lime"
        pred_label, gt_label = "Pred", "GT"

    for c in measure.find_contours(gt_bin, 0.5):
        ax.plot(c[:,1], c[:,0], color=gt_color, linewidth=2, label=gt_label)
    for c in measure.find_contours(pred_bin, 0.5):
        ax.plot(c[:,1], c[:,0], color=pred_color, linewidth=2, label=pred_label)

    # dedupe legend
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=8, frameon=True)

    ax.set_title(f"Contours (thr={thr})"); ax.axis("off")

    plt.suptitle(title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_contour_slices(img_5d, tgt_5d, prob_5d, out_dir, thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    img = img_5d.detach().cpu().numpy()[0,0]
    tgt = tgt_5d.detach().cpu().numpy()[0,0]
    prob = prob_5d.detach().cpu().numpy()[0,0]
    D = img.shape[0]
    paths = []

    for z in range(D):
        gt_bin   = (tgt[z]  >= SLICE_GT_THR).astype(np.uint8)
        pred_bin = (prob[z] >= thr).astype(np.uint8)

        # Dice per slice
        inter = (pred_bin & gt_bin).sum()
        denom = pred_bin.sum() + gt_bin.sum()
        slice_dice = (2 * inter / denom) if denom > 0 else 1.0  # define empty/empty as 1.0

        plt.figure(figsize=(3.2,3.2))
        ax = plt.gca()
        ax.imshow(img[z], cmap="gray")

        if GREEN_IS_PRED:
            pred_color, gt_color = "lime", "red"
            pred_label, gt_label = "Pred", "GT"
        else:
            pred_color, gt_color = "red", "lime"
            pred_label, gt_label = "Pred", "GT"

        for c in measure.find_contours(gt_bin, 0.5):
            ax.plot(c[:,1], c[:,0], color=gt_color, linewidth=2, label=gt_label)
        for c in measure.find_contours(pred_bin, 0.5):
            ax.plot(c[:,1], c[:,0], color=pred_color, linewidth=2, label=pred_label)

        if z == 0:
            h, l = ax.get_legend_handles_labels()
            uniq = dict(zip(l, h))
            if uniq:
                ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=7, frameon=True)

        ax.set_title(f"z={z}  Dice={slice_dice:.3f}")
        ax.axis("off")

        fpath = os.path.join(out_dir, f"slice_{z:02d}.png")
        plt.tight_layout(); plt.savefig(fpath, dpi=140); plt.close()
        paths.append(fpath)
    return paths


# ---------------- main ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices")
    ds = LIDCKaggleDataset(data_root, agg=AGG, return_all_masks=False)

    n_train = int(0.8 * len(ds))
    val_ds = Subset(ds, range(n_train, len(ds)))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_resize)

    model = UNet3D(base=16).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    all_indices = list(range(len(val_ds)))
    random.seed(0); random.shuffle(all_indices)
    keep = set(all_indices[:NUM_SAMPLES])

    for i, (img, tgt, meta) in enumerate(val_loader):
        if i not in keep:
            continue
        img, tgt = img.to(device), tgt.to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(img))
        d = dice_score(prob, tgt, thr=THR)
        patient = meta.get("patient", f"case{i}")
        nodule  = meta.get("nodule", "nodule")

        out_dir = "results/pred_samples"
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"{i:04d}_{patient}_{nodule}_dice-{d:.3f}.png")
        title = f"{patient} / {nodule} | Dice={d:.3f}"
        make_preview(img, tgt, prob, out_png, thr=THR, title=title)
        print(f"Saved {out_png}")

        case_dir = os.path.join(out_dir, f"{patient}_{nodule}")
        slice_paths = save_contour_slices(img, tgt, prob, case_dir, thr=THR)
        if MAKE_GIF and slice_paths:
            gif_path = os.path.join(case_dir, "contours.gif")
            imageio.mimsave(gif_path, [imageio.imread(p) for p in slice_paths], duration=GIF_DURATION)
            print(f"Made {gif_path}")

if __name__ == "__main__":
    main()
