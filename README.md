# 🫁 LIDC Project — Lung Nodule Segmentation and Uncertainty

This repository contains a **PyTorch-based pipeline** for working with the **LIDC-IDRI dataset** (Lung Image Database Consortium).  
It focuses on:
- Handling **multiple expert annotations**
- Creating **soft labels** (consensus masks)
- Computing **annotator disagreement maps**
- Training a **3D U-Net** for lung nodule segmentation

---

## 📂 Repository Structure

lidc_project/
│
├── data/ 
│ └── LIDC-IDRI-slices/ # Kaggle dataset
│
├── scripts/ 
│ ├── dataloader.py # Loads CT volumes & 4 radiologist masks
│ ├── test_dataloader.py # Visualizes soft targets & disagreement maps
│ └── train_unet3d.py # Trains a 3D U-Net using soft targets
│
├── results/ # Model checkpoints, logs, and visualizations
│ ├── models/ 
│ └── preview.png # Example visualization output
│
├── lidc-env/ 
│
├── .gitignore # Ignores data, env, cache files
├── requirements.txt 
└── README.md 
