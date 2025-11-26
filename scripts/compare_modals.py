import os, glob, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.models.unet3d import UNet3D
from scripts.metrics import dice_score, iou_score
from data.dataloader import LIDCKaggleDataset

CONFIG = {
    "data_root": os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices"),
    "ckpt_glob": "results/models/unet3d_best_seed*.pth",  # your saved members
    "target_shape": (16, 128, 128),
    "split_seed": 42,      # SAME split as training
    "thr": 0.3,            # same eval threshold
    "base": 16,
    "csv_out": "results/compare_models.csv",
}

def collate_fn(batch):
    b = batch[0]
    img = b["image"].unsqueeze(0).unsqueeze(0)
    tgt = b["target"].unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=CONFIG["target_shape"], mode="trilinear", align_corners=False)
    tgt = F.interpolate(tgt, size=CONFIG["target_shape"], mode="trilinear", align_corners=False)
    return img, tgt

def build_val_loader():
    ds = LIDCKaggleDataset(CONFIG["data_root"], agg="soft", return_all_masks=False)
    n_train = int(0.8 * len(ds))
    idx = np.arange(len(ds))
    rng = np.random.get_state()
    np.random.seed(CONFIG["split_seed"])
    np.random.shuffle(idx)
    np.random.set_state(rng)
    val_ds = Subset(ds, idx[n_train:])
    return DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

@torch.no_grad()
def eval_single(model, loader, device):
    dice_list, iou_list = [], []
    model.eval()
    for img, tgt in loader:
        img, tgt = img.to(device), tgt.to(device)
        prob = torch.sigmoid(model(img))
        dice_list.append(dice_score(prob, tgt, thr=CONFIG["thr"]))
        iou_list.append(iou_score(prob, tgt, thr=CONFIG["thr"]))
    return float(np.mean(dice_list)), float(np.std(dice_list)), float(np.mean(iou_list)), float(np.std(iou_list))

@torch.no_grad()
def eval_ensemble(models, loader, device):
    dice_list, iou_list = [], []
    for img, tgt in loader:
        img, tgt = img.to(device), tgt.to(device)
        probs = [torch.sigmoid(m(img)) for m in models]
        p_mean = torch.stack(probs, dim=0).mean(0)
        dice_list.append(dice_score(p_mean, tgt, thr=CONFIG["thr"]))
        iou_list.append(iou_score(p_mean, tgt, thr=CONFIG["thr"]))
    return float(np.mean(dice_list)), float(np.std(dice_list)), float(np.mean(iou_list)), float(np.std(iou_list))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(glob.glob(CONFIG["ckpt_glob"]))
    assert ckpts, f"No checkpoints found with: {CONFIG['ckpt_glob']}"

    print("Found checkpoints:")
    for c in ckpts: print(" -", c)

    val_loader = build_val_loader()

    rows = []
    models = []
    for c in ckpts:
        m = UNet3D(base=CONFIG["base"]).to(device)
        state = torch.load(c, map_location=device)
        m.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
        d_mean, d_std, j_mean, j_std = eval_single(m, val_loader, device)
        rows.append(("individual", os.path.basename(c), d_mean, d_std, j_mean, j_std))
        models.append(m)

    # ensemble
    e_d_mean, e_d_std, e_j_mean, e_j_std = eval_ensemble(models, val_loader, device)
    rows.append(("ensemble", f"{len(models)} members", e_d_mean, e_d_std, e_j_mean, e_j_std))

    # pretty print
    print("\n=== Comparison (3D volume metrics on same val split) ===")
    print(f"{'type':11} {'name':30} {'Dice(mean±std)':>18} {'IoU(mean±std)':>18}")
    for t, n, dm, ds, jm, js in rows:
        print(f"{t:11} {n:30} {dm:6.4f}±{ds:5.4f} {jm:12.4f}±{js:5.4f}")

    # save CSV
    os.makedirs(os.path.dirname(CONFIG["csv_out"]), exist_ok=True)
    with open(CONFIG["csv_out"], "w") as f:
        f.write("type,name,dice_mean,dice_std,iou_mean,iou_std\n")
        for t, n, dm, ds, jm, js in rows:
            f.write(f"{t},{n},{dm:.6f},{ds:.6f},{jm:.6f},{js:.6f}\n")
    print(f"\nSaved CSV → {CONFIG['csv_out']}")

if __name__ == "__main__":
    main()
