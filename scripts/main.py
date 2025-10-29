
import os
import torch
from scripts.train_seg import train_segmentation
# or (preferred inside a package)
# from .train_seg import train_segmentation

# ---------------------------
# Configuration (centralized)
# ---------------------------
CONFIG = {
    "data_root": os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices"),
    "epochs": 20,
    "lr": 1e-3,
    "base": 16,                     # UNet width multiplier
    "deterministic": False,         # set True for reproducibility (slower)
    "seed": 666,                    # random seed for reproducibility
    "wandb_project": "lidc_project",
    "wandb_name": "3d_unet_softlabels",
    #"wandb_key": "1201b98d49797282cef724fed65c90effa1bbb0e",  # your API key
}

def set_deterministic(seed: int):
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

    # optional: attach seed to run name
    run_name = f"{CONFIG['wandb_name']}_seed{CONFIG['seed']}"

    # call training function
    train_segmentation(
        data_root=CONFIG["data_root"],
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
        base=CONFIG["base"],
        seed=CONFIG["seed"],
        wandb_project=CONFIG["wandb_project"],
        wandb_name=run_name,
        
    )

if __name__ == "__main__":
    main()
