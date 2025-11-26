#!/usr/bin/env python
"""
viz_soft_au_eu_annotators_full_volume.py
→ Single figure (16×4 grid) showing all slices of a target LIDC case
"""

import os, sys, glob, numpy as np, torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR     = os.path.join(PROJECT_ROOT, "results", "models")
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data", "LIDC-IDRI-slices")
TARGET_SHAPE = (16, 128, 128)
TARGET_PATIENT = "LIDC-IDRI-0284"
TARGET_NODULE  = "nodule-0"
SPLIT_SEED = 666

sys.path.append(PROJECT_ROOT)
from scripts.models.unet3d import UNet3D
from data.dataloader import LIDCKaggleDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def bernoulli_entropy(p, eps=1e-8):
    p = torch.clamp(p, eps, 1-eps)
    return -(p*torch.log(p)+(1-p)*torch.log(1-p))

@torch.no_grad()
def ensemble_forward(models, img):
    return torch.stack([torch.sigmoid(m(img)) for m in models], dim=0)

def resize_3d(x, tgt):
    x = x.unsqueeze(0).unsqueeze(0).float()
    x = F.interpolate(x, size=tgt, mode="trilinear", align_corners=False)
    return x.squeeze(0).squeeze(0)

def resize_masks(m, tgt):
    m = m.unsqueeze(1).float()
    m = F.interpolate(m, size=tgt, mode="trilinear", align_corners=False)
    return m.squeeze(1)

# ---- load ensemble ----
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR,"unet3d_best_seed*.pth")))
models=[]
for p in ckpts:
    m=UNet3D(base=16).to(device)
    s=torch.load(p,map_location=device)
    m.load_state_dict(s["state_dict"] if "state_dict" in s else s)
    m.eval(); models.append(m)

# ---- load dataset ----
ds=LIDCKaggleDataset(root=DATA_ROOT,agg="soft",return_all_masks=True,normalize=True)
n_total=len(ds); n_train=int(0.8*n_total)
idx=np.arange(n_total); np.random.seed(SPLIT_SEED); np.random.shuffle(idx)
val_idx=idx[n_train:]

target=None
for i in val_idx:
    s=ds[i]; meta=s["meta"]
    if meta["patient"]==TARGET_PATIENT and meta["nodule"]==TARGET_NODULE:
        target=s; break
if target is None:
    raise RuntimeError("Target not found in val set")

img,soft,dis,masks=target["image"],target["target"],target["disagreement"],target["masks"]
img=resize_3d(img,TARGET_SHAPE); soft=resize_3d(soft,TARGET_SHAPE)
masks=resize_masks(masks,TARGET_SHAPE)

img_t=img.unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    stack=ensemble_forward(models,img_t)
    p_mean=stack.mean(0)
    alea=bernoulli_entropy(stack).mean(0)
    epi=bernoulli_entropy(p_mean)-alea

img_np,soft_np=img.cpu().numpy(),soft.cpu().numpy()
alea_np,epi_np=alea.squeeze().cpu().numpy(),epi.squeeze().cpu().numpy()
masks_np=masks.cpu().numpy()
D=img_np.shape[0]
colors=["red","lime","cyan","magenta"]

# ---- build tall figure ----
fig,axes=plt.subplots(D,4,figsize=(12,1.5*D))
for z in range(D):
    # 0: CT+annotators
    ax=axes[z,0]
    ax.imshow(img_np[z],cmap="gray",origin="lower")
    for a in range(masks_np.shape[0]):
        ax.contour(masks_np[a,z],levels=[0.5],colors=[colors[a%len(colors)]],linewidths=0.8)
    ax.set_ylabel(f"z={z}",rotation=0,labelpad=15,va="center",fontsize=8)
    ax.axis("off")
    # 1: soft
    im=ax=axes[z,1]; im.imshow(soft_np[z],origin="lower"); ax.axis("off")
    # 2: AU
    im=ax=axes[z,2]; im.imshow(alea_np[z],origin="lower"); ax.axis("off")
    # 3: EU
    im=ax=axes[z,3]; im.imshow(epi_np[z],origin="lower"); ax.axis("off")

axes[0,0].set_title("CT + annotators")
axes[0,1].set_title("Soft label (mean)")
axes[0,2].set_title("Aleatoric (AU)")
axes[0,3].set_title("Epistemic (EU)")
fig.suptitle(f"{TARGET_PATIENT} / {TARGET_NODULE} – All slices ({D})", fontsize=11, y=0.995)
plt.subplots_adjust(
    wspace=0.01,   # horizontal spacing between columns
    hspace=0.05    # vertical spacing between rows
)


out=os.path.join(CKPT_DIR,f"uncertainty_full_{TARGET_PATIENT}_{TARGET_NODULE}.png".replace("/","_"))
fig.savefig(out,dpi=200)
plt.close(fig)
print(f"[INFO] Saved single multi-slice figure to {out}")
