"""Data analysis and visualization module for the Dry Bean dataset."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import FEATURE_COLS, TARGET_COL, FIG_DIR
from data_loader import load_raw_data
from preprocess import _standardize_class_label, _clean_features, EXPECTED_CLASSES


def analyze_data(save=True):
    """Run full data analysis and produce visualizations."""
    print("=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)

    train_df, test_df, val_df = load_raw_data()
    # Fix BOM
    for df in [train_df, test_df, val_df]:
        df.columns = [c.lstrip("﻿") for c in df.columns]

    # Detect dirtiness BEFORE cleaning
    print("\n--- Data Quality Report (Before Cleaning) ---")

    # 1. Check class label dirtiness
    all_labels = pd.concat([train_df[TARGET_COL], test_df[TARGET_COL], val_df[TARGET_COL]])
    raw_unique = sorted(all_labels.unique())
    print(f"  Raw unique class labels: {len(raw_unique)} (expected 7)")
    print(f"  Raw labels: {raw_unique}")

    dirty_labels = [l for l in raw_unique if str(l).strip().upper() not in EXPECTED_CLASSES]
    dirty_count = all_labels[all_labels.apply(lambda x: str(x).strip().upper() not in EXPECTED_CLASSES)].count()
    print(f"  Corrupted labels: {dirty_count} samples across types: {dirty_labels}")

    # 2. Missing values
    all_df_raw = pd.concat([train_df, test_df, val_df], ignore_index=True)
    missing = all_df_raw.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n  Missing values (NaN):")
    for col, cnt in missing.items():
        print(f"    {col}: {cnt}")

    # 3. Non-numeric markers
    solidity_q = all_df_raw["Solidity"].apply(lambda x: str(x).strip() == "?").sum()
    print(f"\n  Solidity entries marked '?': {solidity_q}")

    compactness_cm = all_df_raw["Compactness"].apply(
        lambda x: "cm" in str(x).lower()
    ).sum()
    print(f"  Compactness entries with 'cm' suffix: {compactness_cm}")

    # 4. Duplicates
    dup_rows = all_df_raw.duplicated().sum()
    print(f"\n  Exact duplicate rows: {dup_rows}")

    # Now clean labels for analysis
    for df in [train_df, test_df, val_df]:
        df[TARGET_COL] = df[TARGET_COL].apply(_standardize_class_label)
    all_df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    # Keep only valid classes for analysis
    all_df = all_df[all_df[TARGET_COL].isin(EXPECTED_CLASSES)]
    all_df = _clean_features(all_df)

    print(f"\n--- Dataset Summary (after label cleaning) ---")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Samples:  {len(all_df)}")
    print(f"  Classes:  {all_df[TARGET_COL].nunique()}")

    # Class distribution
    class_counts = all_df[TARGET_COL].value_counts()
    print(f"\n  Class distribution:")
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt} ({cnt / len(all_df) * 100:.1f}%)")

    # Feature statistics
    print(f"\n--- Feature Statistics ---")
    print(all_df[FEATURE_COLS].describe().to_string())

    # Outlier summary
    print(f"\n--- Outlier Detection (3*IQR) ---")
    total_outliers = 0
    for col in FEATURE_COLS:
        Q1 = all_df[col].quantile(0.25)
        Q3 = all_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR
        n_out = ((all_df[col] < lower) | (all_df[col] > upper)).sum()
        total_outliers += n_out
        if n_out > 0:
            print(f"  {col}: {n_out} outliers ({n_out / len(all_df) * 100:.2f}%)")
    print(f"  Total outlier instances: {total_outliers}")

    if save:
        _make_plots(all_df, class_counts)

    return all_df


def _make_plots(all_df, class_counts):
    """Generate all analysis plots."""
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 11
    colors = sns.color_palette("Set2", n_colors=len(class_counts))

    # ---- Plot 1: Class distribution bar chart ----
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(val), ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution in Dry Bean Dataset", fontsize=14, fontweight="bold")
    ax.set_xlabel("Bean Class")
    ax.set_ylabel("Number of Samples")
    ax.set_xticklabels(class_counts.index, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/01_class_distribution.png")

    # ---- Plot 2: Box plot of all features (raw) ----
    fig, ax = plt.subplots(figsize=(18, 8))
    df_melt = all_df.melt(id_vars=[TARGET_COL], value_vars=FEATURE_COLS,
                          var_name="Feature", value_name="Value")
    if len(df_melt) > 20000:
        df_melt = df_melt.sample(20000, random_state=42)
    sns.boxplot(x="Feature", y="Value", data=df_melt, palette="Set3", ax=ax,
                flierprops=dict(marker=".", alpha=0.3))
    ax.set_title("Feature Distributions (Box Plot - Raw Values)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value (raw scale)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_feature_boxplot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/02_feature_boxplot.png")

    # ---- Plot 3: Correlation heatmap ----
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = all_df[FEATURE_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/03_correlation_heatmap.png")

    # ---- Plot 4: Feature histograms per class (key features) ----
    key_features = ["Area", "Perimeter", "MajorAxisLength", "Eccentricity",
                    "Compactness", "ShapeFactor1"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax_i, feat in zip(axes.flat, key_features):
        for cls in sorted(all_df[TARGET_COL].unique()):
            subset = all_df[all_df[TARGET_COL] == cls][feat].dropna()
            if len(subset) > 0:
                ax_i.hist(subset, bins=40, alpha=0.4, label=cls, density=True)
        ax_i.set_title(feat, fontweight="bold")
    axes.flat[0].legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_feature_hist_by_class.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/04_feature_hist_by_class.png")


if __name__ == "__main__":
    analyze_data()
