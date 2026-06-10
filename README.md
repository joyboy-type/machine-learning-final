# Dry Bean Classification - Machine Learning Pipeline

A complete machine learning project for multi-class dry bean classification using geometric shape features.

## Features

- **Data cleaning pipeline**: Handles dirty labels (leet-speak, case variants), missing values ("?", NaN), format errors ("cm" suffix), duplicates, and outliers
- **4 classification algorithms**: SVM (RBF), Random Forest, XGBoost, MLP (Multi-layer Perceptron)
- **Comprehensive evaluation**: Accuracy, Precision, Recall, F1-score, confusion matrix, loss curves, robustness analysis, overfitting analysis
- **Modular CLI tool**: Command-line interface for flexible experimentation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run.py --all

# Run specific algorithm
python run.py --algorithm svm

# Run multiple algorithms
python run.py --algorithm svm,xgboost,mlp

# Data analysis only
python run.py --analysis-only

# Enable PCA dimensionality reduction
python run.py --algorithm all --pca
```

## Results Summary

| Algorithm | Test Accuracy | F1 (Weighted) | Inference (ms/sample) |
|-----------|--------------|---------------|----------------------|
| SVM (RBF) | 93.42% | 93.43% | 0.0473 |
| XGBoost | 92.85% | 92.86% | 0.0030 |
| MLP | 92.41% | 92.43% | 0.0006 |
| Random Forest | 92.33% | 92.34% | 0.0097 |

## Project Structure

```
├── config.py          # Path configuration
├── data_loader.py     # Data loading interface
├── analysis.py        # Data analysis & visualization
├── preprocess.py      # Data cleaning & feature engineering
├── train.py           # Model registry & training
├── evaluate.py        # Evaluation metrics & plots
├── utils.py           # Utility functions
├── run.py             # CLI entry point
└── data/              # Dataset directory
```

## Adding New Models

Use the `@register` decorator in `train.py`:

```python
@register("my_model")
def build_my_model():
    return MyClassifier(...), is_iterative
```
