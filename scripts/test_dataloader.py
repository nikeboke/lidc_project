import os
import matplotlib.pyplot as plt
from dataloader import LIDCKaggleDataset

DATA_ROOT = os.environ.get("LIDC_ROOT", "data/LIDC-IDRI-slices")

ds = LIDCKaggleDataset(DATA_ROOT, agg="soft", return_all_masks=True)
print("Num items:", len(ds))

sample = ds[0]
img = sample["image"]           # (D,H,W)
tgt = sample["target"]          # (D,H,W) soft/consensus
var = sample["disagreement"]    # (D,H,W)
masks = sample["masks"]         # (A,D,H,W)

print("image:", img.shape, img.dtype)
print("target:", tgt.shape, tgt.dtype)
print("disagreement:", var.shape, var.dtype)
print("masks:", masks.shape, masks.dtype, "annotators =", masks.shape[0])

mid = img.shape[0] // 2

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("CT"); plt.imshow(img[mid], cmap="gray"); plt.axis("off")
plt.subplot(1,3,2); plt.title("Soft target"); plt.imshow(img[mid], cmap="gray"); plt.imshow(tgt[mid], alpha=0.4); plt.axis("off")
plt.subplot(1,3,3); plt.title("Disagreement (var)"); plt.imshow(var[mid]); plt.axis("off")
os.makedirs("results", exist_ok=True)
plt.tight_layout()
plt.savefig("results/preview.png", dpi=150)
print("Saved results/preview.png")
