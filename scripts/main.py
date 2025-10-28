# scripts/main.py
import os
import torch
from scripts.models.unet3d import train_segmentation  # training entrypoint

# ---------------------------
# Embedded config (no argparse)
# ---------------------------
CONFIG = {
    "data_root": os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices"),
    "epochs": 20,
    "lr": 1e-3,
    "base": 16,           # UNet width multiplier
    "deterministic": False,
    "seed": 666,
}

def set_deterministic(seed: int = 666):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def main():
    if CONFIG["deterministic"]:
        set_deterministic(CONFIG["seed"])

    train_segmentation(
        data_root=CONFIG["data_root"],
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
        base=CONFIG["base"],
    )

if __name__ == "__main__":
    main()
