# scripts/train_seg.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import wandb
# scripts/train_seg.py
from .metrics import soft_dice_loss_from_logits, dice_score, iou_score
from .models.unet3d import UNet3D
from data.dataloader import LIDCKaggleDataset   # keep this as-is (top-level 'data/')



def train_segmentation(
    data_root: str,
    epochs: int = 20,
    lr: float = 1e-3,
    base: int = 16,
    seed: int | None = None,
    wandb_project: str = "lidc_project",
    wandb_name: str = "3d_unet_softlabels",
):
    # ------------------ setup ------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ------------------ 🟣 WandB setup ------------------
    wandb.login(key='1201b98d49797282cef724fed65c90effa1bbb0e')
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={"agg": "soft", "epochs": epochs, "lr": lr, "seed": seed, "base": base},
    )
    # ---------------------------------------------------

    # ------------------ dataset + dataloader ------------------
    ds = LIDCKaggleDataset(data_root, agg="soft", return_all_masks=False)
    n_train = int(0.8 * len(ds))
    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    train_ds = Subset(ds, indices[:n_train])
    val_ds = Subset(ds, indices[n_train:])

    def collate_fn(batch):
        b = batch[0]
        img = b["image"].unsqueeze(0).unsqueeze(0)
        tgt = b["target"].unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img, size=(16, 128, 128), mode="trilinear", align_corners=False)
        tgt = F.interpolate(tgt, size=(16, 128, 128), mode="trilinear", align_corners=False)
        return img, tgt

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ------------------ model + optimizer ------------------
    model = UNet3D(base=base).to(device)
    pos_weight = torch.tensor([20.0], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ------------------ training loop ------------------
    best_dice = 0.0
    os.makedirs("results/models", exist_ok=True)
    ckpt_path = "results/models/unet3d_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for img, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            img, tgt = img.to(device), tgt.to(device)
            logits = model(img)
            loss = 0.5 * bce(logits, tgt) + 0.5 * soft_dice_loss_from_logits(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # validation
        model.eval()
        val_dice, val_iou = 0.0, 0.0
        with torch.no_grad():
            for img, tgt in val_loader:
                img, tgt = img.to(device), tgt.to(device)
                prob = torch.sigmoid(model(img))
                val_dice += dice_score(prob, tgt, thr=0.3)
                val_iou += iou_score(prob, tgt, thr=0.3)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch {epoch}: loss={tr_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")
        wandb.log({"epoch": epoch, "train_loss": tr_loss, "val_dice": val_dice, "val_iou": val_iou})

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ Saved best model (Dice={best_dice:.4f})")

    wandb.log({"best_dice": best_dice})
    wandb.finish()
