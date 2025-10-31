import os, datetime, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from scripts.train_seg import train_segmentation
from scripts.metrics import dice_score, iou_score
from scripts.models.unet3d import UNet3D
from data.dataloader import LIDCKaggleDataset

CONFIG = {
    "data_root": os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices"),
    "epochs": 20,
    "lr": 1e-3,
    "base": 16,
    "split_seed": 42,                     # ✅ fixed split (identical train/val for all)
    "init_seeds": [11, 22, 33, 44, 55],   # different initialiyation weights 
    "wandb_project": "lidc_project",
    "wandb_entity": "nen_ai",
    "target_shape": (16, 128, 128),
    "thr": 0.3,
}

@torch.no_grad()
def bernoulli_entropy(p, eps=1e-8):
    # H(p) = -p log p - (1-p) log (1-p)
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

def collate_fn_factory(target_shape):
    def _collate(batch):
        b = batch[0]
        img = b["image"].unsqueeze(0).unsqueeze(0)
        tgt = b["target"].unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img, size=target_shape, mode="trilinear", align_corners=False)
        tgt = F.interpolate(tgt, size=target_shape, mode="trilinear", align_corners=False)
        return img, tgt
    return _collate

def build_val_loader(data_root, split_seed, target_shape):
    ds = LIDCKaggleDataset(data_root, agg="soft", return_all_masks=False)
    n_train = int(0.8 * len(ds))
    idx = np.arange(len(ds))
    rng = np.random.get_state()
    np.random.seed(split_seed)     # ✅ identical split
    np.random.shuffle(idx)
    np.random.set_state(rng)
    val_ds = Subset(ds, idx[n_train:])
    return DataLoader(val_ds, batch_size=1, shuffle=False,
                      collate_fn=collate_fn_factory(target_shape))

def load_models(ckpt_paths, base=16, device="cuda"):
    models = []
    for c in ckpt_paths:
        m = UNet3D(base=base).to(device)
        state = torch.load(c, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            m.load_state_dict(state["state_dict"])
        else:
            m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models

def ensemble_forward(models, img):
    """
    Forward all K models and stack probabilities: [K,1,D,H,W]

    Ensemble math (for Bernoulli voxel-wise predictions):
      - Predictive mean:  p̄ = (1/K) Σ_k σ(f_k(x))
      - Total uncertainty:   H(p̄)
      - Aleatoric (data):    E_k[ H( p_k ) ]
      - Epistemic (BALD):    MI = H(p̄) - E_k[ H( p_k ) ]
    """
    with torch.no_grad():
        probs = []
        for m in models:
            logits = m(img)
            probs.append(torch.sigmoid(logits))
        return torch.stack(probs, dim=0)  # [K,1,D,H,W]

# ---------------- Main ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    ensemble_group = f"ensemble_run_{ts}"

    # Train each member (same split_seed, different init seeds)
    for i, init_seed in enumerate(CONFIG["init_seeds"], start=1):
        print(f"\n===== Training ensemble member {i}/{len(CONFIG['init_seeds'])} "
              f"(init_seed={init_seed}, split_seed={CONFIG['split_seed']}) =====")

        run_name = f"unet3d_seed{init_seed}"
        ckpt_path = train_segmentation(
            data_root=CONFIG["data_root"],
            epochs=CONFIG["epochs"],
            lr=CONFIG["lr"],
            base=CONFIG["base"],
            seed=init_seed,                         #  weight init seed
            split_seed=CONFIG["split_seed"],        #  split seed
            wandb_project=CONFIG["wandb_project"],
            wandb_entity=CONFIG["wandb_entity"],
            wandb_group=ensemble_group,
            wandb_name=run_name,
        )

        # Rename checkpoint so each member is kept
        dst = f"results/models/unet3d_best_seed{init_seed}.pth"
        if os.path.exists(ckpt_path):
            if os.path.abspath(ckpt_path) != os.path.abspath(dst):
                os.replace(ckpt_path, dst)
            print(f"✓ Saved {dst}")
        else:
            print(f"⚠️ Warning: expected checkpoint not found at {ckpt_path}")

    # ---------------- Ensemble summary (evaluation + W&B logging) ----------------
    print("\n===== Ensemble summary evaluation =====")
    ckpts = sorted(glob.glob("results/models/unet3d_best_seed*.pth"))
    assert ckpts, "No ensemble checkpoints found (results/models/unet3d_best_seed*.pth)."

    # Init a short W&B run to store the summary
    wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
    wandb.init(
        project=CONFIG["wandb_project"],
        entity=CONFIG["wandb_entity"],
        name=f"ensemble_summary_{ts}",
        group=ensemble_group,
        job_type="ensemble_summary",
        tags=["ensemble", "summary"],
        reinit=True,
        config={
            "thr": CONFIG["thr"],
            "base": CONFIG["base"],
            "split_seed": CONFIG["split_seed"],
            "members": len(ckpts),
        },
    )

    # build the validation loader with the SAME split
    val_loader = build_val_loader(CONFIG["data_root"], CONFIG["split_seed"], CONFIG["target_shape"])

    # load models
    models = load_models(ckpts, base=CONFIG["base"], device=device)
    print(f"Loaded {len(models)} ensemble members:")
    for p in ckpts: print(" -", p)

    # evaluate: Dice/IoU of ensemble mean prediction
    dice_list, iou_list = [], []
    for img, tgt in tqdm(val_loader, desc="Ensemble eval"):
        img, tgt = img.to(device), tgt.to(device)

        # Stack probs across members: [K,1,D,H,W]
        prob_stack = ensemble_forward(models, img)

        # --- Ensemble computations (voxel-wise) ---
        # Predictive mean: p̄ = (1/K) Σ_k p_k
        p_mean = prob_stack.mean(dim=0)
        # Aleatoric: E_k[H(p_k)]
        alea = bernoulli_entropy(prob_stack).mean(dim=0)
        # Total uncertainty: H(p̄)
        p_entropy = bernoulli_entropy(p_mean)
        # Epistemic (BALD): MI = H(p̄) - E_k[H(p_k)]
        mi = p_entropy - alea
        # ------------------------------------------

        d = dice_score(p_mean, tgt, thr=CONFIG["thr"])
        j = iou_score(p_mean, tgt, thr=CONFIG["thr"])
        dice_list.append(d); iou_list.append(j)

    mean_dice, std_dice = float(np.mean(dice_list)), float(np.std(dice_list))
    mean_iou,  std_iou  = float(np.mean(iou_list)),  float(np.std(iou_list))
    print(f"\nEnsemble Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Ensemble IoU : {mean_iou :.4f} ± {std_iou :.4f}")

    wandb.log({
        "ensemble/val_dice_mean": mean_dice,
        "ensemble/val_dice_std": std_dice,
        "ensemble/val_iou_mean": mean_iou,
        "ensemble/val_iou_std": std_iou,
        "ensemble/members": len(models),
    })
    wandb.finish()

if __name__ == "__main__":
    main()
