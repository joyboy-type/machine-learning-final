# Dry Bean Classification — Full-Stack Machine Learning Pipeline

> **2026 AIT209 Machine Learning & Project Practice · Final Project**
>
> A complete ML pipeline for classifying 7 varieties of dry beans using geometric shape features extracted from computer vision.

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-green)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Overview

This project implements an end-to-end machine learning workflow on the **Dry Bean Dataset**, covering data quality analysis, data cleaning, feature engineering, multi-algorithm experiments, and a modular CLI system. The dataset was artificially "polluted" to simulate real-world data quality issues.

### Key Highlights

- **4 classification algorithms**: SVM (RBF), Random Forest, XGBoost, MLP
- **5 evaluation dimensions**: accuracy, per-class F1, loss curves, inference speed, robustness
- **Dual noise robustness**: Gaussian noise + Salt-and-Pepper noise
- **7→25→7**: Automated detection and correction of 171 leet-speak labels, case variants, and whitespace corruption
- **12+ visualization charts**: confusion matrices, robustness curves, loss curves, per-class analysis, overfitting analysis
- **Modular CLI**: `python run.py --all` for one-click execution

---

## Dataset Description

The Dry Bean Dataset contains geometric shape features extracted from images of 13,611 dry bean grains across 7 varieties.

| Attribute | Value |
|-----------|-------|
| **Features** | 16 numeric geometric descriptors (Area, Perimeter, Eccentricity, etc.) |
| **Classes** | 7 bean varieties: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| **Training set** | 9,527 samples (raw) |
| **Test set** | 2,737 samples (raw) |
| **Validation set** | 1,347 samples (raw) |

### Data Contamination (Simulated)

The dataset was intentionally "polluted" to mimic real-world issues:

| Contamination Type | Description | Affected |
|--------------------|-------------|:--------:|
| **Leet-speak labels** | D3RMAS0N (E→3, O→0), H0R0Z, S3K3R, B0MBAY | 171 labels |
| **Case variants** | "dermason", "HOROZ", "cali" → inconsistent casing | ~200 labels |
| **Trailing whitespace** | "BARBUNYA ", "SIRA " | ~200 labels |
| **Missing values** | NaN + "?" markers in Perimeter and Solidity columns | 1,065 cells |
| **Format errors** | " cm" suffix in Compactness column | 388 values |
| **Duplicate rows** | Exact copies across all features | 49 rows |

### Class Distribution

| Class | Samples | % |
|-------|:-------:|:--:|
| DERMASON | 3,546 | 26.1% |
| SIRA | 2,636 | 19.4% |
| SEKER | 2,027 | 14.9% |
| HOROZ | 1,928 | 14.2% |
| CALI | 1,630 | 12.0% |
| BARBUNYA | 1,322 | 9.7% |
| **BOMBAY** | **522** | **3.8%** |

> **Imbalanced dataset**: BOMBAY is under-represented (3.8%). Evaluation uses both macro and weighted averaging.

---

## Data Processing Pipeline

The preprocessing pipeline cleans the data in 5 sequential steps:

| Step | Method | Description |
|:----:|--------|-------------|
| 1 | **Label Standardization** | Leet-speak mapping (D3RMAS0N→DERMASON), case unification, whitespace stripping. Recovers 25 raw labels → 7 correct classes |
| 2 | **Feature Type Repair** | "?"→NaN in Solidity, strip " cm" suffix in Compactness, enforce float64 |
| 3 | **Missing Value Removal** | Drop rows with NaN (~5% in Perimeter, ~2.8% in Solidity) |
| 4 | **Deduplication** | Remove 49 exact duplicate rows |
| 5 | **Outlier Clipping** | 3×IQR clipping per feature (retain samples, suppress extremes) |

### Feature Engineering

- **Z-score Standardization** (StandardScaler): Required for SVM (RBF kernel) and MLP. Chosen over Min-Max for better outlier tolerance.
- **Optional PCA**: `--pca` flag enables PCA with 95% variance retention. Off by default to preserve interpretability.

### Cleaning Statistics

| Dataset | Raw | After Cleaning | Removed |
|---------|:---:|:---:|:---:|
| Train | 9,527 | 8,581 | 946 (9.93%) |
| Test | 2,737 | 2,477 | 260 (9.50%) |
| Validation | 1,347 | 1,226 | 121 (8.98%) |
| **Total** | **13,611** | **12,284** | **1,327 (9.75%)** |

---

## Algorithms

| # | Algorithm | Type | In Class? | Key Params |
|---|-----------|------|:---------:|------------|
| 1 | **SVM (RBF)** | Support Vector Machine | Yes | C=10, gamma=scale |
| 2 | **Random Forest** | Ensemble (Bagging) | Yes | n=200, depth=15 |
| 3 | **XGBoost** | Gradient Boosting | **No** (self-taught) | n=200, lr=0.1, depth=6 |
| 4 | **MLP** | Multi-Layer Perceptron | Partially | layers=(256,128,64), ReLU, Adam |

> **XGBoost** is the "beyond-classroom" algorithm — implemented by studying the original paper (Chen & Guestrin, 2016) and official documentation.

---

## Evaluation Dimensions

### 1. Test Set Accuracy

| Algorithm | Accuracy | F1 (Macro) | F1 (Weighted) | Precision (Macro) | Recall (Macro) |
|-----------|:--------:|:----------:|:-------------:|:-----------------:|:--------------:|
| **SVM** | **0.9342** | **0.9423** | **0.9343** | **0.9447** | **0.9401** |
| XGBoost | 0.9285 | 0.9387 | 0.9286 | 0.9400 | 0.9375 |
| MLP | 0.9241 | 0.9329 | 0.9243 | 0.9355 | 0.9311 |
| Random Forest | 0.9233 | 0.9327 | 0.9234 | 0.9342 | 0.9314 |

### 2. Overfitting Analysis

| Algorithm | Train Acc | Test Acc | Gap |
|-----------|:---------:|:--------:|:---:|
| SVM | 0.9378 | 0.9342 | 0.0036 |
| MLP | 0.9256 | 0.9241 | **0.0015** |
| Random Forest | 0.9913 | 0.9233 | 0.0680 |
| XGBoost | **0.9999** | 0.9285 | 0.0714 |

> **SVM and MLP generalize best**. Tree models slightly overfit despite regularization.

### 3. Inference Speed

| Algorithm | ms/sample | Relative |
|-----------|:---------:|:--------:|
| **MLP** | **0.0006** | 1× |
| XGBoost | 0.0030 | 5× |
| Random Forest | 0.0105 | 18× |
| SVM | 0.0494 | 82× |

### 4. Robustness (Dual Noise Types)

Accuracy under increasing noise levels:

**Gaussian Noise:**
| Noise Level | SVM | XGBoost | MLP | RF |
|:-----------:|:---:|:-------:|:---:|:--:|
| 0.00 | 0.9342 | 0.9285 | 0.9241 | 0.9233 |
| 0.01 | 0.9342 | 0.9298 | 0.9237 | 0.9217 |
| 0.05 | 0.9354 | 0.9294 | 0.9233 | 0.9269 |
| 0.10 | 0.9314 | 0.9233 | 0.9197 | 0.9229 |

**Salt-and-Pepper Noise:**
SVM maintains the best absolute accuracy under all noise types. Both noise types produce consistent degradation patterns.

### 5. Loss Curves (Iterative Models)

XGBoost and MLP provide per-iteration loss tracking:
- **XGBoost**: mlogloss drops from ~1.2 to near 0; validation loss plateaus after ~100 iterations
- **MLP**: Staircase descent with adaptive learning rate triggering at ~40 epochs; final validation score ~0.92

---

## Project Architecture

```
machine-learning-final/
+-- config.py            # Global paths, constants, feature lists
+-- data_loader.py       # Unified data loading (raw/clean/processed)
+-- analysis.py          # Data quality analysis + 4 EDA charts
+-- preprocess.py        # 5-step cleaning pipeline + StandardScaler + optional PCA
+-- train.py             # Model registry (@register decorator) + training interface
+-- evaluate.py          # Full evaluation: metrics, robustness, per-class, charts
+-- utils.py             # Gaussian/SP noise generation, inference timing
+-- run.py               # CLI entry point (argparse)
+-- data/                # Dataset directory (place CSV files here)
+-- output/              # Generated results (auto-created)
    +-- figures/         # 12+ visualization PNGs
    +-- results/         # JSON formatted results
```

### Design Patterns

- **@register decorator** in `train.py` — add new algorithms without modifying any other module
- **Single-responsibility modules** — each `.py` file handles one concern
- **Pipeline automation** — `python run.py --all` from raw data to all evaluation charts in one command

### CLI Examples

```bash
# Full pipeline: analysis → preprocessing → train all 4 models → evaluate
python run.py --all

# Single algorithm training + evaluation
python run.py --algorithm svm

# Multiple specific algorithms
python run.py --algorithm xgboost,mlp,rf

# Data analysis visualizations only
python run.py --analysis-only

# Preprocessing only
python run.py --preprocess-only

# Enable PCA dimensionality reduction
python run.py --algorithm all --pca

# Show help
python run.py --help
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- macOS/Linux/Windows

### Installation

```bash
# Clone the repository
git clone https://github.com/joyboy-type/machine-learning-final.git
cd machine-learning-final

# Install dependencies
pip install -r requirements.txt

# Place the dataset CSV files in the data/ directory:
#   Dry_Bean_Dataset_Dirty_train.csv
#   Dry_Bean_Dataset_Dirty_test.csv
#   Dry_Bean_Dataset_Dirty_val.csv
# (Or update paths in config.py)

# Run the full pipeline
python run.py --all
```

### Adding a New Algorithm

1. Define a builder function in `train.py`:

```python
@register("catboost")
def build_catboost():
    return CatBoostClassifier(iterations=200, depth=6, verbose=False), True
```

2. Use it immediately:

```bash
python run.py --algorithm catboost
```

No changes needed in `run.py` or `evaluate.py`.

---

## Results & Deliverables

All results are generated in the `output/` directory:

| Directory | Contents |
|-----------|----------|
| `output/figures/` | 12+ PNG charts: class distribution, boxplots, correlation heatmap, 4 confusion matrices, metrics comparison, dual robustness curves, loss curves, overfitting analysis, per-class F1 |
| `output/results/` | `all_results.json` — complete structured experiment data |
| `output/*.csv` | Cleaned datasets (train_clean.csv, test_clean.csv, val_clean.csv) |
| `output/*.npz` | Processed NumPy arrays (standardized, ready for training) |

### Paper

The project includes a comprehensive academic paper (Chinese) with cover page, scoring rubric, and detailed experimental analysis. See:

- `论文_DryBean机器学习项目报告.pdf` — Full paper (15 pages, PDF)
- `论文_DryBean机器学习项目报告.md` — Markdown source

---

## License

MIT License — feel free to use and adapt for educational purposes.

---

*Project completed for AIT209 Machine Learning & Project Practice, 2026.*
