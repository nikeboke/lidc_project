
import os, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from scripts.metrics import dice_score, iou_score
from scripts.models.unet3d import UNet3D
from data.dataloader import LIDCKaggleDataset

# ---------------- Config ----------------
CONFIG = {
    "data_root": os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices"),
    "ckpt_glob": "results/models/unet3d_best_seed*.pth",
    "target_shape": (16, 128, 128),
    "thr": 0.3,
    "out_dir": "results/ensemble_eval",
    "num_cases": None,  
}

def bernoulli_entropy(p, eps=1e-8):
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

@torch.no_grad()
def ensemble_forward(models, img):
    probs = []
    for m in models:
        logits = m(img)
        probs.append(torch.sigmoid(logits))
    return torch.stack(probs, dim=0)  # [K,1,D,H,W]

def collate_fn(batch):
    b = batch[0]
    img = b["image"].unsqueeze(0).unsqueeze(0)
    tgt = b["target"].unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=CONFIG["target_shape"], mode="trilinear", align_corners=False)
    tgt = F.interpolate(tgt, size=CONFIG["target_shape"], mode="trilinear", align_corners=False)
    meta = b.get("meta", {})
    return img, tgt, meta

def quick_preview(z_img, z_mean, z_mi, z_alea, fpath, title=""):
    plt.figure(figsize=(12,3))
    ax = plt.subplot(1,4,1); ax.imshow(z_img, cmap="gray"); ax.set_title("CT"); ax.axis("off")
    ax = plt.subplot(1,4,2); ax.imshow(z_mean); ax.set_title("p̄ mean prob"); ax.axis("off")
    ax = plt.subplot(1,4,3); ax.imshow(z_mi); ax.set_title("Epistemic MI"); ax.axis("off")
    ax = plt.subplot(1,4,4); ax.imshow(z_alea); ax.set_title("Aleatoric E[H(p)]"); ax.axis("off")
    plt.suptitle(title)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    plt.tight_layout(); plt.savefig(fpath, dpi=140); plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LIDCKaggleDataset(CONFIG["data_root"], agg="soft", return_all_masks=False)
    n_train = int(0.8 * len(ds))
    val_ds = Subset(ds, range(n_train, len(ds)))
    if CONFIG["num_cases"] is not None:
        val_ds = Subset(val_ds, list(range(min(CONFIG["num_cases"], len(val_ds)))))

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    ckpts = sorted(glob.glob(CONFIG["ckpt_glob"]))
    assert ckpts, f"No checkpoints found matching {CONFIG['ckpt_glob']}"
    print(f"Loading {len(ckpts)} ensemble members:")
    for c in ckpts: print(" -", c)

    models = []
    for c in ckpts:
        m = UNet3D(base=16).to(device)
        state = torch.load(c, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            m.load_state_dict(state["state_dict"])
        else:
            m.load_state_dict(state)
        m.eval()
        models.append(m)

    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    dice_list, iou_list = [], []

    for i, (img, tgt, meta) in enumerate(tqdm(val_loader, desc="Ensemble eval")):
        img, tgt = img.to(device), tgt.to(device)
        prob_stack = ensemble_forward(models, img)          # [K,1,D,H,W]
        p_mean = prob_stack.mean(dim=0)                     # predictive mean
        alea = bernoulli_entropy(prob_stack).mean(dim=0)    # aleatoric
        p_entropy = bernoulli_entropy(p_mean)               # total entropy
        mi = p_entropy - alea                               # epistemic (BALD)

        d = dice_score(p_mean, tgt, thr=CONFIG["thr"])
        j = iou_score(p_mean, tgt, thr=CONFIG["thr"])
        dice_list.append(d); iou_list.append(j)

        patient = meta.get("patient", f"case{i}")
        nodule  = meta.get("nodule", "nodule")
        base = os.path.join(CONFIG["out_dir"], f"{i:04d}_{patient}_{nodule}")

        np.savez_compressed(
            base + ".npz",
            p_mean=p_mean.detach().cpu().numpy()[0,0],
            epistemic_mi=mi.detach().cpu().numpy()[0,0],
            aleatoric=alea.detach().cpu().numpy()[0,0],
            target=tgt.detach().cpu().numpy()[0,0],
            ct=img.detach().cpu().numpy()[0,0],
            dice=d, iou=j,
        )

        z = img.shape[2] // 2
        quick_preview(
            z_img=img.detach().cpu().numpy()[0,0,z],
            z_mean=p_mean.detach().cpu().numpy()[0,0,z],
            z_mi=mi.detach().cpu().numpy()[0,0,z],
            z_alea=alea.detach().cpu().numpy()[0,0,z],
            fpath=base + "_preview.png",
            title=f"{patient}/{nodule} | Dice={d:.3f} IoU={j:.3f}",
        )

    print(f"\nEnsemble Dice: mean={np.mean(dice_list):.4f} ± {np.std(dice_list):.4f}")
    print(f"Ensemble IoU : mean={np.mean(iou_list):.4f} ± {np.std(iou_list):.4f}")

if __name__ == "__main__":
    main()
