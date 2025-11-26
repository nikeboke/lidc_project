#!/usr/bin/env python
"""
explore_rawdata_stats.py

RAW DATA STATS using the SAME annotator counting function as in
eval_uncertainty_correlations_and_viz.py:

    n_annotators = count_annotators_from_masks(masks)

Outputs (in results/rawdata_stats/):

CSV:
  - rawdata_cases.csv       (one row per nodule)
      patient, nodule, D, H, W, n_annotators
  - rawdata_patients.csv    (one row per patient)
      patient, total_slices, n_nodules, n_annotators_patient
  - rawdata_summary.csv     (global stats)

Plots:
  - hist_slices_per_case.png
  - box_slices_per_case.png
  - hist_slices_per_patient_total.png
  - hist_nodules_per_patient.png
  - bar_n_annotators_per_patient.png
"""

import os, sys, collections, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- project paths ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "LIDC-IDRI-slices")
OUT_DIR   = os.path.join(PROJECT_ROOT, "results", "rawdata_stats")
os.makedirs(OUT_DIR, exist_ok=True)

from data.dataloader import LIDCKaggleDataset


# ---------- SAME COUNTING FUNCTION AS IN EVAL SCRIPT ----------
def count_annotators_from_masks(masks_vol, thr: float = 0.5) -> int:
    """
    Count annotators by checking which masks actually contain any positive voxels.

    masks_vol: torch.Tensor or np.ndarray with shape [A, D, H, W]
    thr: threshold for "positive" (0/1 masks → 0.5 is fine)
    """
    if hasattr(masks_vol, "detach"):  # torch tensor
        m = masks_vol.detach().cpu().numpy()
    else:
        m = np.asarray(masks_vol)

    m_bin = m > thr                       # [A,D,H,W] -> bool
    has_pos = m_bin.reshape(m_bin.shape[0], -1).any(axis=1)  # [A]
    return int(has_pos.sum())


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[CSV] Saved {len(rows)} rows to {path}")


def main():
    print("[INFO] PROJECT_ROOT:", PROJECT_ROOT)
    print("[INFO] DATA_ROOT   :", DATA_ROOT)
    print("[INFO] OUT_DIR     :", OUT_DIR)

    # -------- load dataset (no split, we look at all cases) --------
    ds = LIDCKaggleDataset(
        root=DATA_ROOT,
        agg="soft",
        return_all_masks=True,
        normalize=True,
    )

    n_total = len(ds)
    print(f"[INFO] Total cases (nodules) in dataset: {n_total}")

    case_records = []   # one row per nodule
    patients_set = set()

    # -------- per-case stats --------
    for idx in range(n_total):
        sample = ds[idx]
        img   = sample["image"]    # [D,H,W]
        masks = sample["masks"]    # [A,D,H,W]
        meta  = sample["meta"]     # contains 'patient', 'nodule'

        patient = meta["patient"]
        nodule  = meta["nodule"]
        patients_set.add(patient)

        D, H, W = img.shape

        # n_annotators: USE THE SAME FUNCTION AS IN EVAL SCRIPT
        n_annotators = count_annotators_from_masks(masks, thr=0.5)

        case_records.append({
            "patient": patient,
            "nodule": nodule,
            "D": int(D),
            "H": int(H),
            "W": int(W),
            "n_annotators": int(n_annotators),
        })

    n_patients = len(patients_set)
    n_cases    = len(case_records)
    print(f"[INFO] n_patients = {n_patients}")
    print(f"[INFO] n_cases    = {n_cases}")

    # -------- slices per case --------
    slices_per_case = np.array([r["D"] for r in case_records], dtype=np.int32)
    print("\n=== SLICES PER CASE (nodule) ===")
    print(f"min  : {int(slices_per_case.min())}")
    print(f"max  : {int(slices_per_case.max())}")
    print(f"mean : {float(slices_per_case.mean()):.2f}")
    case_quantiles = {}
    for q in [0, 25, 50, 75, 90, 100]:
        val = np.percentile(slices_per_case, q)
        case_quantiles[q] = float(val)
        print(f"{q:3.0f}th percentile: {val:.2f}")

    # -------- per-patient aggregates --------
    by_patient = collections.defaultdict(list)
    for r in case_records:
        by_patient[r["patient"]].append(r)

    patient_records = []
    total_slices_per_patient = []
    nodules_per_patient      = []
    n_annotators_per_patient = []

    for patient, rows in by_patient.items():
        total_D = int(sum(r["D"] for r in rows))
        n_nod   = int(len(rows))
        # patient-level annotators = max over their nodules
        max_ann = int(max(r["n_annotators"] for r in rows))

        total_slices_per_patient.append(total_D)
        nodules_per_patient.append(n_nod)
        n_annotators_per_patient.append(max_ann)

        patient_records.append({
            "patient": patient,
            "total_slices": total_D,
            "n_nodules": n_nod,
            "n_annotators_patient": max_ann,
        })

    total_slices_per_patient = np.array(total_slices_per_patient, dtype=np.int32)
    nodules_per_patient      = np.array(nodules_per_patient, dtype=np.int32)
    n_annotators_per_patient = np.array(n_annotators_per_patient, dtype=np.int32)

    # --- total slices per patient ---
    print("\n=== TOTAL SLICES PER PATIENT (sum over nodules) ===")
    print(f"min  : {int(total_slices_per_patient.min())}")
    print(f"max  : {int(total_slices_per_patient.max())}")
    print(f"mean : {float(total_slices_per_patient.mean()):.2f}")
    patient_slices_quantiles = {}
    for q in [0, 25, 50, 75, 90, 100]:
        val = np.percentile(total_slices_per_patient, q)
        patient_slices_quantiles[q] = float(val)
        print(f"{q:3.0f}th percentile: {val:.2f}")

    # --- nodules per patient ---
    print("\n=== NODULES PER PATIENT ===")
    print(f"min  : {int(nodules_per_patient.min())}")
    print(f"max  : {int(nodules_per_patient.max())}")
    print(f"mean : {float(nodules_per_patient.mean()):.2f}")
    nodules_quantiles = {}
    for q in [0, 25, 50, 75, 90, 100]:
        val = np.percentile(nodules_per_patient, q)
        nodules_quantiles[q] = float(val)
        print(f"{q:3.0f}th percentile: {val:.2f}")

    # --- value counts: n_annotators per patient ---
    print("\n=== VALUE COUNTS: N_ANNOTATORS PER PATIENT (max over nodules) ===")
    counter_ann = collections.Counter(n_annotators_per_patient.tolist())
    for k in sorted(counter_ann.keys()):
        print(f"{k} annotators: {counter_ann[k]} patients")

    # -------- CSVs --------

    # 1) per-case CSV
    cases_csv = os.path.join(OUT_DIR, "rawdata_cases.csv")
    write_csv(
        cases_csv,
        fieldnames=["patient", "nodule", "D", "H", "W", "n_annotators"],
        rows=case_records,
    )

    # 2) per-patient CSV
    patients_csv = os.path.join(OUT_DIR, "rawdata_patients.csv")
    write_csv(
        patients_csv,
        fieldnames=["patient", "total_slices", "n_nodules", "n_annotators_patient"],
        rows=patient_records,
    )

    # 3) summary CSV
    summary_rows = []
    summary_rows.append({"metric": "n_patients", "value": n_patients})
    summary_rows.append({"metric": "n_cases",    "value": n_cases})

    summary_rows.append({"metric": "slices_per_case_min",  "value": float(slices_per_case.min())})
    summary_rows.append({"metric": "slices_per_case_max",  "value": float(slices_per_case.max())})
    summary_rows.append({"metric": "slices_per_case_mean", "value": float(slices_per_case.mean())})
    for q, val in case_quantiles.items():
        summary_rows.append({"metric": f"slices_per_case_p{int(q)}", "value": val})

    summary_rows.append({"metric": "total_slices_per_patient_min",  "value": float(total_slices_per_patient.min())})
    summary_rows.append({"metric": "total_slices_per_patient_max",  "value": float(total_slices_per_patient.max())})
    summary_rows.append({"metric": "total_slices_per_patient_mean", "value": float(total_slices_per_patient.mean())})
    for q, val in patient_slices_quantiles.items():
        summary_rows.append({"metric": f"total_slices_per_patient_p{int(q)}", "value": val})

    summary_rows.append({"metric": "nodules_per_patient_min",  "value": float(nodules_per_patient.min())})
    summary_rows.append({"metric": "nodules_per_patient_max",  "value": float(nodules_per_patient.max())})
    summary_rows.append({"metric": "nodules_per_patient_mean", "value": float(nodules_per_patient.mean())})
    for q, val in nodules_quantiles.items():
        summary_rows.append({"metric": f"nodules_per_patient_p{int(q)}", "value": val})

    for k in sorted(counter_ann.keys()):
        summary_rows.append({
            "metric": f"n_patients_with_{k}_annotators",
            "value": int(counter_ann[k]),
        })

    summary_csv = os.path.join(OUT_DIR, "rawdata_summary.csv")
    write_csv(
        summary_csv,
        fieldnames=["metric", "value"],
        rows=summary_rows,
    )

    # -------- plots --------

    # 1) histogram of slices per case
    plt.figure(figsize=(6,4))
    plt.hist(slices_per_case, bins=20)
    plt.xlabel("Slices per case (nodule)")
    plt.ylabel("Count")
    plt.title("Distribution of slices per case")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "hist_slices_per_case.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {out_path}")

    # 2) boxplot of slices per case
    plt.figure(figsize=(4,5))
    plt.boxplot(slices_per_case, vert=True, showfliers=True)
    plt.ylabel("Slices per case (nodule)")
    plt.title("Boxplot of slices per case")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "box_slices_per_case.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {out_path}")

    # 3) histogram of total slices per patient
    plt.figure(figsize=(6,4))
    plt.hist(total_slices_per_patient, bins=20)
    plt.xlabel("Total slices per patient")
    plt.ylabel("Count")
    plt.title("Distribution of total slices per patient")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "hist_slices_per_patient_total.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {out_path}")

    # 4) histogram of nodules per patient
    plt.figure(figsize=(6,4))
    plt.hist(
        nodules_per_patient,
        bins=range(int(nodules_per_patient.min()),
                   int(nodules_per_patient.max()) + 2)
    )
    plt.xlabel("Nodules per patient")
    plt.ylabel("Count")
    plt.title("Distribution of nodules per patient")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "hist_nodules_per_patient.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {out_path}")

    # 5) bar plot of n_annotators per patient
    keys_sorted = sorted(counter_ann.keys())
    counts      = [counter_ann[k] for k in keys_sorted]

    plt.figure(figsize=(6,4))
    plt.bar(keys_sorted, counts)
    plt.xlabel("Number of annotators (per patient, max over nodules)")
    plt.ylabel("Number of patients")
    plt.title("Annotator counts per patient")
    plt.xticks(keys_sorted)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "bar_n_annotators_per_patient.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {out_path}")

    print("\n[INFO] Done. Check CSVs + PNGs in:", OUT_DIR)


if __name__ == "__main__":
    main()
