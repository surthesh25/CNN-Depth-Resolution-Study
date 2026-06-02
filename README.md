# CNN Image Classification — AIT 636 Assignment 13(B)

> Exploring how **CNN depth** and **image resolution** affect classification accuracy across two image datasets.

---

## Overview

This project trains Convolutional Neural Networks (CNNs) with 1, 2, and 3 convolutional blocks across three image resolutions (32×32, 64×64, 112×112) on two datasets:

| Dataset | Classes | Task |
|---|---|---|
| **Cat–Dog** | 2 | Binary classification |
| **Visual Domain Decathlon (Subset)** | 10 | Multi-class classification |

This produces **9 model configurations per dataset** (3 depths × 3 image sizes), giving a clear picture of how architecture depth and input resolution trade off against each other.

---

## CNN Architecture

Each model follows the same pattern, scaled by depth (`j = 1, 2, 3`):

```
Input (img_size × img_size × 3)
  │
  ├─ [Block 1]  Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
  ├─ [Block 2]  Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)   ← j ≥ 2
  └─ [Block 3]  Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)   ← j = 3
  │
  Flatten → Dense(128, relu) → Dropout(0.5) → Dense(num_classes)
```

**Enhancements over baseline:**
- `BatchNormalization` after every conv layer — stabilises training
- `Dropout(0.25)` in conv blocks + `Dropout(0.5)` in dense head — reduces overfitting
- `EarlyStopping` (patience=4) — stops training when val_accuracy plateaus
- `ReduceLROnPlateau` — halves learning rate when val_loss stalls
- Accuracy **and** loss curves saved per run
- Results heatmap for quick visual comparison across all 9 configurations

---

## Project Structure

```
cnn-image-classification/
│
├── src/
│   └── train.py            # Full experiment script
│
├── data/                   # ← Place your datasets here (not committed)
│   ├── cat-dog/
│   │   ├── train/
│   │   │   ├── cat/
│   │   │   └── dog/
│   │   └── test/
│   │       ├── cat/
│   │       └── dog/
│   └── vdd/
│       ├── train/
│       │   ├── 0001/ … 0010/
│       └── test/
│           ├── 0001/ … 0010/
│
├── outputs/                # Generated plots & CSVs (created at runtime)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your data

Unzip your Cat-Dog and VDD datasets into `data/` following the folder structure above (one sub-folder per class inside `train/` and `test/`).

### 3. Run with default paths

```bash
python src/train.py
```

### 4. Run with custom data paths

```bash
python src/train.py \
  --catdog_train "path/to/cat-dog/train" \
  --catdog_test  "path/to/cat-dog/test" \
  --vdd_train    "path/to/vdd/train" \
  --vdd_test     "path/to/vdd/test" \
  --epochs 20 \
  --batch_size 128
```

All plots and a results CSV are saved to `outputs/`.

---

## Outputs

For each run the script produces:

| File | Description |
|---|---|
| `outputs/<dataset>/samples_img<N>.png` | 5×5 sample grid at each resolution |
| `outputs/<dataset>/curves_img<N>_depth<j>.png` | Train/val accuracy & loss curves |
| `outputs/<dataset>/heatmap_accuracy.png` | 3×3 accuracy heatmap across all configs |
| `outputs/results_<timestamp>.csv` | Full numeric results table |

---

## Key Concepts Demonstrated

- Effect of **input resolution** on feature richness and training time
- Effect of **CNN depth** on representational capacity
- Role of **BatchNormalization** and **Dropout** in regularisation
- **Early stopping** to prevent overfitting and wasted compute
- Systematic **grid experiment** design across hyperparameter combinations

---

## Author

**Surthesh Velu Samy**  
AIT 636 — Interpretable Machine Learning
