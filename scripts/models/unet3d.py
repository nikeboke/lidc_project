import os, random, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
from dataloader import LIDCKaggleDataset   # uses agg="soft"

# -------- 3D U-Net (minimal, single-channel in an out 

#Architecture: Lightweight 3D U-Net

#Loss: 0.5 × BCE + 0.5 × Soft Dice

#Labels: Soft-probability masks (average of 4 expert annotations)

#Uncertainty: Voxel-wise variance of masks = annotator disagreement   it models aleatoric uncertainty that’s already in the data.

# 🔐 Login to W&B
wandb.login(key='1201b98d49797282cef724fed65c90effa1bbb0e')

wandb.init(
    project="lidc_project",
    name="3d_unet_softlabels",
    config={"agg": "soft", "epochs": 20, "lr": 1e-3}
)

# ---------------- 3D UNet ----------------
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

# ---------------- Metrics ----------------
def dice_score(y_pred, y_true, eps=1e-6):
    y_pred = (y_pred > 0.5).float()
    num = 2 * (y_pred * y_true).sum()
    den = y_pred.sum() + y_true.sum() + eps
    return (num / den).item()

def iou_score(y_pred, y_true, eps=1e-6):
    y_pred = (y_pred > 0.5).float()
    inter = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - inter + eps
    return (inter / union).item()

# ---------------- Dataloader ----------------
def collate_fn(batch):
    b = batch[0]
    img = b["image"].unsqueeze(0).unsqueeze(0)
    tgt = b["target"].unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=(64,128,128), mode="trilinear", align_corners=False)
    tgt = F.interpolate(tgt, size=(64,128,128), mode="trilinear", align_corners=False)
    return img, tgt

# ---------------- Train ----------------
def train_segmentation(data_root, epochs=20, lr=1e-3, base=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LIDCKaggleDataset(data_root, agg="soft", return_all_masks=False)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = Subset(ds, range(0, n_train)), Subset(ds, range(n_train, len(ds)))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = UNet3D(base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    wandb.init(project="lidc_3d_segmentation", name="unet3d_dice_iou",
               config={"epochs": epochs, "lr": lr, "base": base, "agg": "soft"})

    best_dice = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0
        for img, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            img, tgt = img.to(device), tgt.to(device)
            logits = model(img)
            loss = 0.5 * bce(logits, tgt) + 0.5 * (1 - dice_score(torch.sigmoid(logits), tgt))
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # Validation
        model.eval()
        val_dice, val_iou = 0, 0
        with torch.no_grad():
            for img, tgt in val_loader:
                img, tgt = img.to(device), tgt.to(device)
                prob = torch.sigmoid(model(img))
                val_dice += dice_score(prob, tgt)
                val_iou += iou_score(prob, tgt)
        val_dice /= len(val_loader)
        val_iou  /= len(val_loader)

        print(f"Epoch {epoch}: loss={tr_loss:.4f}, val Dice={val_dice:.4f}, val IoU={val_iou:.4f}")
        wandb.log({"epoch": epoch, "train_loss": tr_loss, "val_dice": val_dice, "val_iou": val_iou})

        if val_dice > best_dice:
            best_dice = val_dice
            os.makedirs("results/models", exist_ok=True)
            torch.save(model.state_dict(), "results/models/unet3d_best.pth")
            print(f"✓ Saved best model (Dice={best_dice:.4f})")

    wandb.log({"best_dice": best_dice})
    wandb.finish()

# ---------------- Main ----------------
if __name__ == "__main__":
    data_root = os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices")
    train_segmentation(data_root)
