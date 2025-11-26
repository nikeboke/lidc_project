#!/usr/bin/env python
"""
viz_soft_au_eu.py

Visualize, for a single LIDC validation case:

  - Soft label (mean of annotators)
  - Model aleatoric uncertainty (expected entropy over ensemble)
  - Model epistemic uncertainty (mutual information / BALD)

All shown as 2D slices side-by-side.

Usage example:

  /home/boeke/Desktop/lidc_project/lidc-env/bin/python \
    /home/boeke/Desktop/lidc_project/scripts/viz_soft_au_eu.py \
    --ckpt_dir /home/boeke/Desktop/lidc_project/results/models \
    --data_root /home/boeke/Desktop/lidc_project/data/LIDC-IDRI-slices \
    --case_idx 0
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
import matplotlib.pyplot as plt

# make local imports work when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.models.unet3d import UNet3D  # type: ignore
from data.dataloader import LIDCKaggleDataset  # type: ignore

CONFIG = {
    "target_shape": (16, 128, 128),
    "split_seed": 666,
}


@torch.no_grad()
def bernoulli_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Elementwise Bernoulli entropy for probabilities in [0,1]."""
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def collate_fn_factory(target_shape: Tuple[int, int, int]):
    """
    Collate: adds batch/channel dims and resizes image, target, disagreement.
    We only need target + meta for this visualization, but reuse same collate.
    """
    def _collate(batch):
        b = batch[0]
        img = b["image"].unsqueeze(0).unsqueeze(0)          # [1,1,D,H,W]
        tgt = b["target"].unsqueeze(0).unsqueeze(0)        # [1,1,D,H,W]
        dis = b["disagreement"].unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

        img = F.interpolate(img, size=target_shape, mode="trilinear",
                            align_corners=False)
        tgt = F.interpolate(tgt, size=target_shape, mode="trilinear",
                            align_corners=False)
        dis = F.interpolate(dis, size=target_shape, mode="trilinear",
                            align_corners=False)

        return img, tgt, dis, b["meta"]
    return _collate


def build_val_subset(data_root: str,
                     split_seed: int,
                     target_shape: Tuple[int, int, int]) -> Subset:
    ds = LIDCKaggleDataset(
        root=data_root,
        agg="soft",
        return_all_masks=False,
        normalize=True,
    )
    n_train = int(0.8 * len(ds))
    idx = np.arange(len(ds))
    rng = np.random.get_state()
    np.random.seed(split_seed)
    np.random.shuffle(idx)
    np.random.set_state(rng)

    val_idx = idx[n_train:]
    val_ds = Subset(ds, val_idx)
    return val_ds


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
def ensemble_forward(models: List[torch.nn.Module],
                     img: torch.Tensor) -> torch.Tensor:
    """
    img: [1,1,D,H,W]
    returns: [K,1,D,H,W]
    """
    probs = [torch.sigmoid(m(img)) for m in models]
    return torch.stack(probs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory with unet3d_best_seed*.pth")
    parser.add_argument("--data_root", type=str,
                        default=os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices"))
    parser.add_argument("--case_idx", type=int, default=0,
                        help="Index of validation case to visualize")
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ----- load ensemble -----
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "unet3d_best_seed*.pth")))
    if not ckpts:
        raise RuntimeError(f"No checkpoints matching 'unet3d_best_seed*.pth' in {args.ckpt_dir}")
    print(f"Found {len(ckpts)} ensemble members:")
    for p in ckpts:
        print("  -", p)
    models = load_models(ckpts, base=args.base, device=device)

    # ----- build validation subset -----
    val_ds = build_val_subset(
        data_root=args.data_root,
        split_seed=CONFIG["split_seed"],
        target_shape=CONFIG["target_shape"],
    )
    if args.case_idx < 0 or args.case_idx >= len(val_ds):
        raise IndexError(f"case_idx {args.case_idx} out of range (0..{len(val_ds)-1})")

    # wrap subset in loader to reuse collate
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_factory(CONFIG["target_shape"]),
    )

    # fetch desired case
    for i, (img, tgt, dis, meta) in enumerate(val_loader):
        if i == args.case_idx:
            break
    else:
        raise RuntimeError("Could not fetch case_idx from val_loader")

    img = img.to(device)  # [1,1,D,H,W]
    tgt = tgt.to(device)  # [1,1,D,H,W]

    with torch.no_grad():
        prob_stack = ensemble_forward(models, img)  # [K,1,D,H,W]
        p_mean = prob_stack.mean(dim=0)            # [1,D,H,W]
        alea = bernoulli_entropy(prob_stack).mean(dim=0)  # [1,D,H,W]
        p_ent = bernoulli_entropy(p_mean)                 # [1,D,H,W]
        epi = p_ent - alea                                # [1,D,H,W]

    soft_np = tgt.squeeze().cpu().numpy()   # [D,H,W]
    alea_np = alea.squeeze().cpu().numpy()  # [D,H,W]
    epi_np = epi.squeeze().cpu().numpy()    # [D,H,W]

    D = soft_np.shape[0]
    mid = D // 2

    patient = meta["patient"]
    nodule = meta["nodule"]
    print(f"Visualizing patient={patient}, nodule={nodule}, slice={mid}/{D}")

    # ----- 3-panel slice visualization -----
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axes[0].imshow(soft_np[mid], origin="lower")
    axes[0].set_title("Soft label (mean of annotators)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(alea_np[mid], origin="lower")
    axes[1].set_title("Aleatoric uncertainty (AU)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(epi_np[mid], origin="lower")
    axes[2].set_title("Epistemic uncertainty (EU)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{patient} / {nodule} – central slice", fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(
        args.ckpt_dir,
        f"soft_au_eu_slice_{patient}_{nodule}.png".replace("/", "_")
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved 3-panel visualization to {out_path}")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    main()
