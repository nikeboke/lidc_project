import os, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

from dataloader import LIDCKaggleDataset

# --- must match training ---
TARGET_SHAPE = (16, 128, 128)   # D,H,W
AGG          = "soft"           # labels: soft or vote2, etc.
CKPT_PATH    = "results/models/unet3d_best.pth"

# ---------------- 3D U-Net (same as train) ----------------
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1), nn.InstanceNorm3d(out_c), nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1), nn.InstanceNorm3d(out_c), nn.ReLU(inplace=True),
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
    img = img_5d.cpu().numpy()[0,0]     # (D,H,W)
    tgt = tgt_5d.cpu().numpy()[0,0]
    prob = prob_5d.cpu().numpy()[0,0]
    z = img.shape[0] // 2
    pred_bin = (prob[z] >= thr).astype(np.float32)

    plt.figure(figsize=(12,3))
    ax = plt.subplot(1,4,1); ax.imshow(img[z], cmap="gray"); ax.set_title("CT"); ax.axis("off")
    ax = plt.subplot(1,4,2); ax.imshow(img[z], cmap="gray"); ax.imshow(tgt[z], alpha=0.4); ax.set_title("GT overlay"); ax.axis("off")
    ax = plt.subplot(1,4,3); im = ax.imshow(prob[z]); ax.set_title("Pred prob"); ax.axis("off")
    ax = plt.subplot(1,4,4); ax.imshow(img[z], cmap="gray"); ax.imshow(pred_bin, alpha=0.4); ax.set_title(f"Pred > {thr}"); ax.axis("off")
    plt.suptitle(title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    data_root = os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices")
    ds = LIDCKaggleDataset(data_root, agg=AGG, return_all_masks=False)

    # match the split you used in training (last 20% as val)
    n_train = int(0.8 * len(ds))
    val_ds = Subset(ds, range(n_train, len(ds)))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_resize)

    # load model
    model = UNet3D(base=16).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # sample a few items
    idxs = list(range(len(val_ds)))
    random.seed(0); random.shuffle(idxs)
    idxs = idxs[:args.num_samples]

    saved = []
    for i, (img, tgt, meta) in enumerate(val_loader):
        if i not in idxs:  # cheap filter; DataLoader is sequential
            continue
        img, tgt = img.to(device), tgt.to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(img))
        d = dice_score(prob, tgt, thr=args.thr)
        patient = meta.get("patient", "patient")
        nodule  = meta.get("nodule", "nodule")
        out_dir = "results/pred_samples"
        fname = f"{i:04d}_{patient}_{nodule}_dice-{d:.3f}.png"
        out_png = os.path.join(out_dir, fname)
        make_preview(img, tgt, prob, out_png, thr=args.thr,
                     title=f"{patient} / {nodule} | Dice={d:.3f}")
        print(f"saved {out_png}")
        saved.append(out_png)

    # optional: log to W&B if available
    try:
        import wandb
        wandb.log({"pred_samples": [wandb.Image(p) for p in saved]})
    except Exception:
        pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
