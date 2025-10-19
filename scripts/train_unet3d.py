import os, random, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader import LIDCKaggleDataset   # uses agg="soft"

# -------- 3D U-Net (minimal, single-channel in/out) --------
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

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)   # logits
# -----------------------------------------------------------

def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2.0 * (prob * target).sum()
    den = (prob.pow(2).sum() + target.pow(2).sum() + eps)
    return 1.0 - num / den

@torch.no_grad()
def dice_score_from_logits(logits, target, thr=0.5, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob >= thr).float()
    num = 2.0 * (pred * target).sum()
    den = (pred.sum() + target.sum() + eps)
    return (num / den).item()

def make_loaders(data_root, val_frac=0.2, seed=0, target_shape=(64,128,128), batch_size=1):
    ds = LIDCKaggleDataset(data_root, agg="soft", return_all_masks=False)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    split = int(len(idx) * (1 - val_frac))
    tr_idx, va_idx = idx[:split], idx[split:]
    train_ds, val_ds = Subset(ds, tr_idx), Subset(ds, va_idx)

    def collate(batch):
        # batch size 1 (volumes differ in shape); resize to target_shape
        b = batch[0]
        img = b["image"].unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        tgt = b["target"].unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img, size=target_shape, mode="trilinear", align_corners=False)
        tgt = F.interpolate(tgt, size=target_shape, mode="trilinear", align_corners=False)
        return img, tgt

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    return train_loader, val_loader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs("results/models", exist_ok=True)

    print(f"Using dataset from: {args.data_root}")
    train_loader, val_loader = make_loaders(args.data_root, val_frac=args.val_frac,
                                            seed=args.seed, target_shape=tuple(args.shape),
                                            batch_size=1)

    model = UNet3D(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best_dice, best_path = 0.0, "results/models/unet3d_best.pth"

    for epoch in range(1, args.epochs+1):
        model.train()
        tbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        tr_loss = 0.0
        for img, tgt in tbar:
            img, tgt = img.to(device), tgt.to(device)
            logits = model(img)
            loss = 0.5 * bce(logits, tgt) + 0.5 * dice_loss_from_logits(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
            tbar.set_postfix(loss=f"{loss.item():.4f}")

        # validation
        model.eval()
        val_dice, val_loss = 0.0, 0.0
        with torch.no_grad():
            for img, tgt in val_loader:
                img, tgt = img.to(device), tgt.to(device)
                logits = model(img)
                val_loss += (0.5 * bce(logits, tgt) + 0.5 * dice_loss_from_logits(logits, tgt)).item()
                val_dice += dice_score_from_logits(logits, tgt)

        n_tr, n_va = len(train_loader), len(val_loader)
        tr_loss /= max(n_tr,1); val_loss /= max(n_va,1); val_dice /= max(n_va,1)
        print(f"Epoch {epoch}: train loss {tr_loss:.4f} | val loss {val_loss:.4f} | val Dice {val_dice:.4f}")

        # save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"state_dict": model.state_dict(),
                        "dice": best_dice,
                        "shape": args.shape}, best_path)
            print(f"✓ Saved new best to {best_path} (Dice={best_dice:.4f})")

    # final save
    final_path = "results/models/unet3d_last.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training done. Best Dice={best_dice:.4f}")
    print(f"Saved: {best_path} and {final_path}")

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape", type=int, nargs=3, default=[64,128,128], help="D H W")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse()
    if not os.path.exists(args.data_root):
        raise SystemExit(f"✗ Dataset path not found: {args.data_root}\n"
                         f"Set LIDC_ROOT or put data under ./data/LIDC-IDRI-slices")
    train(args)
