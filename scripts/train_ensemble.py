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


if __name__ == "__main__" and __package__ is None:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



CONFIG = {
    "data_root": os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices"),
    "epochs": 10,
    "lr": 1e-3,
    "base": 16,
    "split_seed": 666,                     # identical split for all members
    "init_seeds": [101, 202, 303, 404, 505],  # five NEW seeds
    "wandb_project": "lidc_project",
    "wandb_entity": "nen_ai",
    "target_shape": (16, 128, 128),
    "thr": 0.3,
    "train_members": True,                 # force (re)training
}

os.environ.setdefault("WANDB__SERVICE_WAIT", "15")

@torch.no_grad()
def bernoulli_entropy(p, eps=1e-8):
    """Voxel-wise Bernoulli entropy"""
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
    np.random.seed(split_seed)
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
    """Run all ensemble members and stack voxel-wise probabilities."""
    with torch.no_grad():
        probs = [torch.sigmoid(m(img)) for m in models]
        return torch.stack(probs, dim=0)  # [K,1,D,H,W]


# ---------------- Main ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    ensemble_group = f"ensemble_run_{ts}"

    # dedicated subfolder per ensemble run to avoid collisions with past runs
    ckpt_dir = os.path.join("results", "models", ensemble_group)
    os.makedirs(ckpt_dir, exist_ok=True)


    if CONFIG["train_members"]:
        print("🟣 Training ensemble members...")
        for i, init_seed in enumerate(CONFIG["init_seeds"], start=1):
            print(f"\n===== Training member {i}/{len(CONFIG['init_seeds'])} "
                  f"(init_seed={init_seed}, split_seed={CONFIG['split_seed']}) =====")

            run_name = f"unet3d_seed{init_seed}"
            ckpt_path = train_segmentation(
                data_root=CONFIG["data_root"],
                epochs=CONFIG["epochs"],
                lr=CONFIG["lr"],
                base=CONFIG["base"],
                seed=init_seed,                         # weight init seed
                split_seed=CONFIG["split_seed"],        # fixed split for all
                wandb_project=CONFIG["wandb_project"],
                wandb_entity=CONFIG["wandb_entity"],
                wandb_group=ensemble_group,
                wandb_name=run_name,
            )

            # move/rename checkpoint into the per-run folder with the seed in the name
            dst = os.path.join(ckpt_dir, f"unet3d_best_seed{init_seed}.pth")
            if ckpt_path and os.path.exists(ckpt_path):
                if os.path.abspath(ckpt_path) != os.path.abspath(dst):
                    os.replace(ckpt_path, dst)
                print(f"✓ Saved {dst}")
            else:
                print(f"⚠️ Warning: checkpoint not found at {ckpt_path}")
    else:
        print("✅ Skipping training (train_members=False).")

    # =======================
    # (2) Ensemble Evaluation
    # =======================
    print("\n===== Ensemble summary evaluation =====")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "unet3d_best_seed*.pth")))
    assert ckpts, f"No ensemble checkpoints found in {ckpt_dir}."

    # short W&B run to record summary metrics
    try:
        wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
    except Exception:
        pass

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
            "ckpt_dir": ckpt_dir,
        },
        settings=wandb.Settings(save_code=False),
    )

    val_loader = build_val_loader(CONFIG["data_root"], CONFIG["split_seed"], CONFIG["target_shape"])
    models = load_models(ckpts, base=CONFIG["base"], device=device)
    print(f"Loaded {len(models)} ensemble members from {ckpt_dir}:")
    for p in ckpts: print(" -", p)

    # evaluate Dice/IoU for ensemble (3D volume metrics)
    dice_list, iou_list = [], []
    for img, tgt in tqdm(val_loader, desc="Ensemble eval"):
        img, tgt = img.to(device), tgt.to(device)
        prob_stack = ensemble_forward(models, img)
        p_mean = prob_stack.mean(dim=0)
        alea = bernoulli_entropy(prob_stack).mean(dim=0)
        p_entropy = bernoulli_entropy(p_mean)
        mi = p_entropy - alea  # epistemic (BALD), unused here but ready to log/save if needed

        d = dice_score(p_mean, tgt, thr=CONFIG["thr"])
        j = iou_score(p_mean, tgt, thr=CONFIG["thr"])
        dice_list.append(d)
        iou_list.append(j)

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

    # =======================
    # (3) Compare vs individual models (same group)
    # =======================
    print("\n===== Ensemble vs Individual Comparison =====")
    try:
        api = wandb.Api()
        runs = api.runs(f"{CONFIG['wandb_entity']}/{CONFIG['wandb_project']}")

        indiv_best_dice, indiv_best_iou = [], []
        for r in runs:
            if r.group == ensemble_group and not str(r.name).startswith("ensemble_summary"):
                best_d = r.summary.get("best_dice", None)
                best_j = r.summary.get("best_val_iou", r.summary.get("val_iou", None))
                if best_d is not None:
                    indiv_best_dice.append(float(best_d))
                if best_j is not None:
                    indiv_best_iou.append(float(best_j))
                print(f"{r.name}: best_dice={best_d} | best_iou={best_j}")

        if indiv_best_dice:
            mean_indiv_d = float(np.mean(indiv_best_dice))
            std_indiv_d  = float(np.std(indiv_best_dice))
            wandb.log({
                "comparison/individual_best_dice_mean": mean_indiv_d,
                "comparison/individual_best_dice_std":  std_indiv_d,
                "comparison/ensemble_vs_individual_dice_gain": mean_dice - mean_indiv_d,
            })

        if indiv_best_iou:
            mean_indiv_j = float(np.mean(indiv_best_iou))
            std_indiv_j  = float(np.std(indiv_best_iou))
            wandb.log({
                "comparison/individual_best_iou_mean": mean_indiv_j,
                "comparison/individual_best_iou_std":  std_indiv_j,
                "comparison/ensemble_vs_individual_iou_gain": mean_iou - mean_indiv_j,
            })
    except Exception as e:
        print(f"⚠️ W&B comparison fetch failed: {e}")

    try:
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
