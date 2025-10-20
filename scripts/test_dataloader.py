import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure   
from dataloader import LIDCKaggleDataset


DATA_ROOT = os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices")
SAMPLE_IDX = int(os.environ.get("LIDC_SAMPLE", 1000))  # override with LIDC_SAMPLE=...

# --- basic checks ---
if not os.path.exists(DATA_ROOT):
    raise SystemExit(f"✗ Dataset path not found: {DATA_ROOT}\n"
                     f"Set LIDC_ROOT or put data under ./data/LIDC-IDRI-slices")
print(f"Using dataset from: {DATA_ROOT}")

# --- dataset ---
ds = LIDCKaggleDataset(DATA_ROOT, agg="soft", return_all_masks=True)
print("Num items:", len(ds))

# clamp sample index if out of range
if SAMPLE_IDX < 0 or SAMPLE_IDX >= len(ds):
    print(f"Requested sample {SAMPLE_IDX} out of range. Using 0 instead.")
    SAMPLE_IDX = 0

sample = ds[SAMPLE_IDX]
img = sample["image"]           # (D,H,W)
tgt = sample["target"]          # (D,H,W) soft/consensus
var = sample["disagreement"]    # (D,H,W)
masks = sample["masks"]         # (A,D,H,W)

print("image:", img.shape, img.dtype)
print("target:", tgt.shape, tgt.dtype)
print("disagreement:", var.shape, var.dtype)
print("masks:", masks.shape, masks.dtype, "annotators =", masks.shape[0])
print("meta:", sample.get("meta", {}))

mid = img.shape[0] // 2
base = img[mid]
mask_list = masks[:, mid]  # (A,H,W), typically A=4

# colors for annotators (opaque lines)
colors = ["red", "lime", "blue", "yellow"]
labels = [f"Annotator {i+1}" for i in range(mask_list.shape[0])]

plt.figure(figsize=(18, 4))

# 1️⃣ CT slice
plt.subplot(1, 4, 1)
plt.imshow(base, cmap="gray")
plt.title("CT slice")
plt.axis("off")

# 2️⃣ Soft target overlay
plt.subplot(1, 4, 2)
plt.imshow(base, cmap="gray")
plt.imshow(tgt[mid], cmap="Reds", alpha=0.4)
plt.title("Soft target (consensus)")
plt.axis("off")

# 3️⃣ Disagreement map
plt.subplot(1, 4, 3)
plt.imshow(var[mid], cmap="inferno")
plt.title("Expert disagreement (variance)")
plt.axis("off")

# 4️⃣ Annotator boundaries
plt.subplot(1, 4, 4)
plt.imshow(base, cmap="gray")


for i, mask in enumerate(mask_list):
    # ensure mask is a numpy array
    mask_np = mask.detach().cpu().numpy()
    # find contours at the 0.5 level
    contours = measure.find_contours(mask_np, 0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color=colors[i], linewidth=1.5)




# legend with annotator colors
for i, c in enumerate(colors[:mask_list.shape[0]]):
    plt.plot([], [], color=c, linewidth=2, label=labels[i])
plt.legend(loc="lower right", fontsize=8, frameon=True)
plt.title("Annotator mask boundaries")
plt.axis("off")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/preview.png", dpi=150)
print("✓ Saved results/preview.png (boundaries only)")

