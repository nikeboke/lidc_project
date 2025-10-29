import os
import re
from typing import List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio.v3 as iio

def _numsort(files: List[str]) -> List[str]:
    # numeric-aware sort: 0001.png, 0002.png, ...
    def key(s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', s)]
    return sorted(files, key=key)

def _load_volume(dir_path: str) -> np.ndarray:
    files = _numsort([f for f in os.listdir(dir_path) if not f.startswith('.')])
    vol = [iio.imread(os.path.join(dir_path, f)) for f in files]
    vol = np.stack(vol, axis=0)  # (D,H,W)
    return vol

def _minmax(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

class LIDCKaggleDataset(Dataset):
    """
    Expects Kaggle layout:
      LIDC-IDRI-XXXX/<nodule-id>/{images, mask-0, mask-1, mask-2, mask-3}
    """
    def __init__(
        self,
        root: str,
        agg: str = "soft",          # "soft"|"vote2"|"vote3"|"union"|"intersect"
        return_all_masks: bool = False,
        transforms: Optional[callable] = None,
        normalize: bool = True,
    ):
        self.root = root
        self.agg = agg
        self.return_all_masks = return_all_masks
        self.transforms = transforms
        self.normalize = normalize
        self.items = self._index_items()

    def _index_items(self) -> List[Dict]:
        items = []
        patients = [p for p in os.listdir(self.root) if not p.startswith('.')]
        for p in sorted(patients):
            pdir = os.path.join(self.root, p)
            if not os.path.isdir(pdir):
                continue
            # nodules under patient
            for n in sorted(os.listdir(pdir)):
                ndir = os.path.join(pdir, n)
                if not os.path.isdir(ndir):
                    continue
                if not os.path.isdir(os.path.join(ndir, "images")):
                    continue
                # require 4 masks if present
                mask_dirs = [d for d in os.listdir(ndir) if d.startswith("mask")]
                if len(mask_dirs) == 0:
                    continue
                items.append({
                    "patient": p,
                    "nodule": n,
                    "img_dir": os.path.join(ndir, "images"),
                    "mask_dirs": [os.path.join(ndir, f"mask-{k}") for k in range(4) if os.path.isdir(os.path.join(ndir, f"mask-{k}"))]
                })
        if len(items) == 0:
            raise RuntimeError(f"No items found under {self.root}")
        return items

    def __len__(self):
        return len(self.items)

    def _aggregate(self, masks: np.ndarray) -> np.ndarray:
        """
        masks: (A=annotators, D,H,W) binary {0,1}
        returns aggregated target (D,H,W), float32
        """
        A = masks.shape[0]
        if self.agg == "soft":
            return masks.mean(axis=0).astype(np.float32)                       # probability map
        if self.agg == "vote2":
            return (masks.sum(axis=0) >= 2).astype(np.float32)                 # >=2 votes
        if self.agg == "vote3":
            return (masks.sum(axis=0) >= 3).astype(np.float32)
        if self.agg == "union":
            return (masks.sum(axis=0) >= 1).astype(np.float32)
        if self.agg == "intersect":
            return (masks.sum(axis=0) == A).astype(np.float32)
        raise ValueError(f"Unknown agg='{self.agg}'")

    def __getitem__(self, idx: int) -> Dict:
        meta = self.items[idx]
        vol = _load_volume(meta["img_dir"]).astype(np.float32)  # (D,H,W)

        masks_list = []
        for md in meta["mask_dirs"]:
            mvol = _load_volume(md)
            # ensure binary {0,1}
            mvol = (mvol > 0).astype(np.float32)
            masks_list.append(mvol)
        masks = np.stack(masks_list, axis=0)  # (A,D,H,W)

        if self.normalize:
            vol = _minmax(vol)

        target = self._aggregate(masks)      # (D,H,W)
        # disagreement / aleatoric proxy:
        # variance across annotators (higher => more disagreement)
        if masks.shape[0] > 1:
            disagreement = masks.var(axis=0).astype(np.float32)  # (D,H,W)
        else:
            disagreement = np.zeros_like(target, dtype=np.float32)

        sample = {
            "image": torch.from_numpy(vol),                    # (D,H,W)
            "target": torch.from_numpy(target),                # (D,H,W)
            "disagreement": torch.from_numpy(disagreement),    # (D,H,W)
            "meta": {"patient": meta["patient"], "nodule": meta["nodule"]}
        }
        if self.return_all_masks:
            sample["masks"] = torch.from_numpy(masks)          # (A,D,H,W)

        if self.transforms:
            sample = self.transforms(sample)

        return sample
