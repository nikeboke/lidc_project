#!/usr/bin/env python
"""
eval_uncertainty_vs_softlabel.py

Compute how model aleatoric and epistemic uncertainties relate to
soft labels (mean of annotators) in LIDC.

We compute:
  corr(alea_hat, soft_label)
  corr(epi_hat, soft_label)

If disentanglement is meaningful, aleatoric should correlate more
strongly (positively) with soft labels, while epistemic should differ.
"""

import os
import sys
import glob
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# --- make local imports work ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.models.unet3d import UNet3D  # type: ignore
from data.dataloader import LIDCKaggleDataset  # type: ignore

CONFIG = {
    "target_shape": (16, 128, 128),
    "split_seed": 666,
    "max_voxels_per_case": 10000,  # subsample per case for global stats
}


@torch.no_grad()
def bernoulli_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def collate_fn_factory(target_shape: Tuple[int, int, int]):
    def _collate(batch):
        b = batch[0]

        img = b["image"].unsqueeze(0).unsqueeze(0)       # [1,1,D,H,W]
        tgt = b["target"].unsqueeze(0).unsqueeze(0)     # [1,1,D,H,W]
        dis = b["disagreement"].unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

        img = F.interpolate(img, size=target_shape, mode="trilinear", align_corners=False)
        tgt = F.interpolate(tgt, size=target_shape, mode="trilinear", align_corners=False)
        dis = F.interpolate(dis, size=target_shape, mode="trilinear", align_corners=False)

        return img, tgt, dis, b["meta"]
    return _collate


def build_val_loader(data_root: str, split_seed: int, target_shape: Tuple[int, int, int]) -> DataLoader:
    ds = LIDCKaggleDataset(root=data_root, agg="soft", return_all_masks=False)
    n_train = int(0.8 * len(ds))
    idx = np.arange(len(ds))
    rng = np.random.get_state()
    np.random.seed(split_seed)
    np.random.shuffle(idx)
    np.random.set_state(rng)
    val_ds = Subset(ds, idx[n_train:])
    return DataLoader(val_ds, batch_size=1, shuffle=False,
                      collate_fn=collate_fn_factory(target_shape))


def load_models(ckpt_paths: List[str], base: int, device: str) -> List[torch.nn.Module]:
    models = []
    for path in ckpt_paths:
        m = UNet3D(base=base).to(device)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            m.load_state_dict(state["state_dict"])
        else:
            m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def ensemble_forward(models: List[torch.nn.Module], img: torch.Tensor) -> torch.Tensor:
    probs = [torch.sigmoid(m(img)) for m in models]
    return torch.stack(probs, dim=0)  # [K,1,D,H,W]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def mean_std_nan(xs: List[float]) -> Tuple[float, float]:
    arr = np.array(xs, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def append_sampled(src: np.ndarray, dst_list: List[np.ndarray], max_voxels: int) -> None:
    if max_voxels is not None and src.size > max_voxels:
        idx = np.random.choice(src.size, size=max_voxels, replace=False)
        src = src[idx]
    dst_list.append(src)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str,
                        default=os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices"))
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "unet3d_best_seed*.pth")))
    assert ckpts, f"No checkpoints found in {args.ckpt_dir}"
    print(f"Found {len(ckpts)} ensemble members:")
    for c in ckpts: print("  -", c)

    models = load_models(ckpts, base=args.base, device=device)
    val_loader = build_val_loader(args.data_root, CONFIG["split_seed"], CONFIG["target_shape"])

    per_case_corr_alea = []
    per_case_corr_epi = []
    case_ids = []

    all_soft = []
    all_alea = []
    all_epi = []

    print("\nRunning evaluation (corr vs soft label)...\n")
    for img, tgt, dis, meta in val_loader:
        img, tgt = img.to(device), tgt.to(device)

        prob_stack = ensemble_forward(models, img)  # [K,1,D,H,W]
        p_mean = prob_stack.mean(dim=0)
        alea_hat = bernoulli_entropy(prob_stack).mean(dim=0)
        p_entropy = bernoulli_entropy(p_mean)
        epi_hat = p_entropy - alea_hat

        soft_np = tgt.squeeze().cpu().numpy().ravel()
        alea_np = alea_hat.squeeze().cpu().numpy().ravel()
        epi_np = epi_hat.squeeze().cpu().numpy().ravel()

        c_alea = safe_corr(alea_np, soft_np)
        c_epi = safe_corr(epi_np, soft_np)
        per_case_corr_alea.append(c_alea)
        per_case_corr_epi.append(c_epi)
        case_ids.append((meta["patient"], meta["nodule"]))

        print(f"{meta['patient']}/{meta['nodule']}: corr(alea,soft)={c_alea:.4f}, corr(epi,soft)={c_epi:.4f}")

        append_sampled(soft_np, all_soft, CONFIG["max_voxels_per_case"])
        append_sampled(alea_np, all_alea, CONFIG["max_voxels_per_case"])
        append_sampled(epi_np, all_epi, CONFIG["max_voxels_per_case"])

    mA, sA = mean_std_nan(per_case_corr_alea)
    mE, sE = mean_std_nan(per_case_corr_epi)
    print("\n==================== SUMMARY ====================")
    print(f"corr(alea_hat, soft label): {mA:.4f} ± {sA:.4f}")
    print(f"corr(epi_hat,  soft label): {mE:.4f} ± {sE:.4f}")
    print("=================================================")

    soft_all = np.concatenate(all_soft)
    alea_all = np.concatenate(all_alea)
    epi_all = np.concatenate(all_epi)
    gA = safe_corr(alea_all, soft_all)
    gE = safe_corr(epi_all, soft_all)
    print("\nGlobal correlations:")
    print(f"corr(alea_hat, soft label) = {gA:.4f}")
    print(f"corr(epi_hat,  soft label) = {gE:.4f}")

    out_path = os.path.join(args.ckpt_dir, "uncertainty_vs_softlabel.npz")
    np.savez_compressed(out_path,
                        per_case_corr_alea=per_case_corr_alea,
                        per_case_corr_epi=per_case_corr_epi,
                        case_ids=np.array(case_ids, dtype=object),
                        global_corr_alea=gA,
                        global_corr_epi=gE)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
