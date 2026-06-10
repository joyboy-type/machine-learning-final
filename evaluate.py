"""Evaluation module: accuracy, precision, recall, F1, confusion matrix, plots."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from config import FIG_DIR, RESULT_DIR, NOISE_LEVELS
from utils import add_gaussian_noise, time_inference
from data_loader import get_class_names


def evaluate_model(model, X_test, y_test, model_name, class_names=None):
    """Compute all metrics for a trained model. Returns dict."""
    if class_names is None:
        class_names = get_class_names()

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    inf_time = time_inference(model, X_test)

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "inference_time_ms": inf_time,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_names": class_names,
    }


def evaluate_robustness(model, X_test, y_test, model_name, noise_type="gaussian"):
    """Evaluate model accuracy under different noise levels."""
    results = {}
    for level in NOISE_LEVELS:
        if level == 0.0:
            X_noisy = X_test
        elif noise_type == "gaussian":
            X_noisy = add_gaussian_noise(X_test, level)
        else:
            X_noisy = add_gaussian_noise(X_test, level)  # fallback
            # Can add salt_pepper here
        acc = accuracy_score(y_test, model.predict(X_noisy))
        results[level] = acc
    return results


def plot_confusion_matrix(cm, class_names, model_name, save_path=None):
    """Plot and save a confusion matrix heatmap."""
    if save_path is None:
        save_path = f"{FIG_DIR}/cm_{model_name}.png"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_metrics_comparison(all_results, save_path=None):
    """Plot bar chart comparing accuracy, F1, and inference time across models."""
    if save_path is None:
        save_path = f"{FIG_DIR}/metrics_comparison.png"

    models = list(all_results.keys())
    accs = [all_results[m]["accuracy"] for m in models]
    f1s = [all_results[m]["f1_weighted"] for m in models]
    times = [all_results[m]["inference_time_ms"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = sns.color_palette("Set2", n_colors=len(models))

    axes[0].bar(models, accs, color=colors, edgecolor="white")
    axes[0].set_title("Accuracy", fontweight="bold")
    axes[0].set_ylim(0.8, 1.0)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=10)

    axes[1].bar(models, f1s, color=colors, edgecolor="white")
    axes[1].set_title("F1 Score (Weighted)", fontweight="bold")
    axes[1].set_ylim(0.8, 1.0)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=10)

    axes[2].bar(models, times, color=colors, edgecolor="white")
    axes[2].set_title("Inference Time (ms/sample)", fontweight="bold")
    for i, v in enumerate(times):
        axes[2].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)

    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_robustness_curves(robustness_results, save_path=None):
    """Plot robustness curves under increasing noise levels."""
    if save_path is None:
        save_path = f"{FIG_DIR}/robustness_curves.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v"]
    for i, (model_name, results) in enumerate(robustness_results.items()):
        levels = list(results.keys())
        accs = list(results.values())
        ax.plot(levels, accs, marker=markers[i % len(markers)],
                label=model_name.upper(), linewidth=2, markersize=8)
    ax.set_xlabel("Gaussian Noise Level (std multiplier)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Model Robustness Under Gaussian Noise", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_loss_curves(histories, save_path=None):
    """Plot training/validation loss curves for iterative models."""
    if save_path is None:
        save_path = f"{FIG_DIR}/loss_curves.png"

    iterative_models = {k: v for k, v in histories.items() if v is not None}
    if not iterative_models:
        print("  No iterative model histories to plot.")
        return None

    n = len(iterative_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (model_name, hist) in zip(axes, iterative_models.items()):
        train_loss = hist.get("train_loss", [])
        val_loss = hist.get("val_loss", [])
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
        if val_loss is not None and len(val_loss) > 0:
            ax.plot(epochs, val_loss, label="Val Loss", linewidth=2, linestyle="--")
        ax.set_title(f"{model_name.upper()} Loss Curve", fontweight="bold")
        ax.set_xlabel("Iteration / Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_overfitting_analysis(train_accs, test_accs, save_path=None):
    """Plot train vs test accuracy for overfitting analysis."""
    if save_path is None:
        save_path = f"{FIG_DIR}/overfitting_analysis.png"

    models = list(train_accs.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, [train_accs[m] for m in models], width,
                   label="Train Accuracy", color="#5B9BD5", edgecolor="white")
    bars2 = ax.bar(x + width / 2, [test_accs[m] for m in models], width,
                   label="Test Accuracy", color="#ED7D31", edgecolor="white")

    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Test Accuracy (Overfitting Analysis)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.4f}",
                ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.4f}",
                ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_all_results(all_results, all_robustness, train_accs, test_accs, histories):
    """Save all results to JSON."""
    output = {
        "metrics": {},
        "robustness": all_robustness,
        "overfitting": {"train_acc": train_accs, "test_acc": test_accs},
    }
    for model_name, res in all_results.items():
        output["metrics"][model_name] = {
            k: v for k, v in res.items()
            if k not in ("confusion_matrix", "classification_report")
        }
        output["metrics"][model_name]["confusion_matrix"] = res["confusion_matrix"]

    path = f"{RESULT_DIR}/all_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to {path}")
