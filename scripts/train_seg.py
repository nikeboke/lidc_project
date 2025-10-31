import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import wandb

from .metrics import soft_dice_loss_from_logits, dice_score, iou_score
from .models.unet3d import UNet3D
from data.dataloader import LIDCKaggleDataset


def train_segmentation(
    data_root: str,
    epochs: int = 20,
    lr: float = 1e-3,
    base: int = 16,
    # seeds
    seed: int | None = None,            # weight init seed (DIFFERENT per ensemble member)
    split_seed: int | None = None,      # dataset split seed (SAME across ensemble)
    # wandb
    wandb_project: str = "lidc_project",
    wandb_name: str = "3d_unet_softlabels",
    wandb_entity: str = "nen_ai",
    wandb_group: str | None = None,     # group runs (e.g., "ensemble_2025-10-29")
    wandb_key: str | None = None,       
) -> str:
    """
    Trains a 3D U-Net on LIDC slices.

    Returns: path to the best checkpoint saved.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------ seeding ------------------
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ------------------ WandB setup ------------------
    api_key = wandb_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        try:
            wandb.login(key=api_key)
        except Exception:
            # continue without crashing if login is already set in the env/session
            pass
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name,
        group=wandb_group,
        config={
            "agg": "soft",
            "epochs": epochs,
            "lr": lr,
            "seed": seed,               # weight init seed
            "split_seed": split_seed,   # split seed
            "base": base,
        },
        reinit=True,  # allows multiple runs in same process (useful for ensembles)
        job_type="train",
        tags=["3d-unet", "lidc", "soft-labels"],
    )

    
    wandb.define_metric("epoch", hidden=True)               # hide "epoch" panel
    wandb.define_metric("train_loss", step_metric="epoch")  # x-axis = epoch
    wandb.define_metric("val_dice",  step_metric="epoch")
    wandb.define_metric("val_iou",   step_metric="epoch")


    # ------------------ dataset + dataloader ------------------
    # Ensure identical split across ensemble: use split_seed (not weight seed)
    if split_seed is None:
        # allow overriding via env for convenience
        split_seed = int(os.environ.get("SPLIT_SEED", "667"))
    print(f"[INFO] Using split_seed={split_seed} (consistent split across ensemble)")

    ds = LIDCKaggleDataset(data_root, agg="soft", return_all_masks=False)
    n_train = int(0.8 * len(ds))
    indices = np.arange(len(ds))
    rng_state = np.random.get_state()  # preserve any external RNG state
    np.random.seed(split_seed)
    np.random.shuffle(indices)
    np.random.set_state(rng_state)

    train_ds = Subset(ds, indices[:n_train])
    val_ds   = Subset(ds, indices[n_train:])

    def collate_fn(batch):
        b = batch[0]
        img = b["image"].unsqueeze(0).unsqueeze(0)
        tgt = b["target"].unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img, size=(16, 128, 128), mode="trilinear", align_corners=False)
        tgt = F.interpolate(tgt, size=(16, 128, 128), mode="trilinear", align_corners=False)
        return img, tgt

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_fn)

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
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        # validation
        model.eval()
        val_dice, val_iou = 0.0, 0.0
        with torch.no_grad():
            for img, tgt in val_loader:
                img, tgt = img.to(device), tgt.to(device)
                prob = torch.sigmoid(model(img))
                val_dice += dice_score(prob, tgt, thr=0.3)
                val_iou  += iou_score(prob,  tgt, thr=0.3)
        val_dice /= max(1, len(val_loader))
        val_iou  /= max(1, len(val_loader))

        print(f"Epoch {epoch}: loss={tr_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")
        wandb.log({ "train_loss": tr_loss,"val_dice": val_dice, "val_iou": val_iou,}, step=epoch)


        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ Saved best model (Dice={best_dice:.4f})")

    wandb.log({"best_dice": best_dice})
    wandb.finish()

    return ckpt_path
