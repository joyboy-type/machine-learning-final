"""Evaluation module: metrics, confusion matrix, robustness, per-class analysis, plots."""

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
from utils import add_gaussian_noise, add_salt_pepper_noise, time_inference
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

    # Per-class metrics
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, cls_name in enumerate(class_names):
        per_class[cls_name] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int((y_test == i).sum()),
        }

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
        "per_class": per_class,
        "inference_time_ms": inf_time,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_names": class_names,
    }


def evaluate_robustness(model, X_test, y_test, model_name, noise_type="gaussian"):
    """Evaluate model accuracy under different noise levels (gaussian or salt_pepper)."""
    results = {}
    for level in NOISE_LEVELS:
        if level == 0.0:
            X_noisy = X_test
        elif noise_type == "gaussian":
            X_noisy = add_gaussian_noise(X_test, level)
        elif noise_type == "salt_pepper":
            X_noisy = add_salt_pepper_noise(X_test, level)
        else:
            X_noisy = add_gaussian_noise(X_test, level)
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


def plot_metrics_comparison(all_results, train_times=None, save_path=None):
    """Plot bar chart comparing accuracy, F1, inference time, and training time."""
    if save_path is None:
        save_path = f"{FIG_DIR}/metrics_comparison.png"

    models = list(all_results.keys())
    accs = [all_results[m]["accuracy"] for m in models]
    f1s = [all_results[m]["f1_weighted"] for m in models]
    times = [all_results[m]["inference_time_ms"] for m in models]

    n_plots = 4 if train_times and len(train_times) == len(models) else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 3:
        axes = list(axes)

    colors = sns.color_palette("Set2", n_colors=len(models))

    axes[0].bar(models, accs, color=colors, edgecolor="white")
    axes[0].set_title("Test Accuracy", fontweight="bold")
    axes[0].set_ylim(0.85, 1.0)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=10)

    axes[1].bar(models, f1s, color=colors, edgecolor="white")
    axes[1].set_title("F1 Score (Weighted)", fontweight="bold")
    axes[1].set_ylim(0.85, 1.0)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=10)

    axes[2].bar(models, times, color=colors, edgecolor="white")
    axes[2].set_title("Inference Time (ms/sample)", fontweight="bold")
    for i, v in enumerate(times):
        axes[2].text(i, v + max(times) * 0.02, f"{v:.4f}", ha="center", fontsize=10)

    if n_plots == 4 and train_times:
        t_times = [train_times[m] for m in models]
        axes[3].bar(models, t_times, color=colors, edgecolor="white")
        axes[3].set_title("Training Time (s)", fontweight="bold")
        for i, v in enumerate(t_times):
            axes[3].text(i, v + max(t_times) * 0.02, f"{v:.2f}", ha="center", fontsize=10)

    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_robustness_curves(robustness_results_all, save_path=None):
    """Plot robustness curves under Gaussian AND Salt-Pepper noise, side by side."""
    if save_path is None:
        save_path = f"{FIG_DIR}/robustness_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax_idx, (noise_name, noise_label) in enumerate([
        ("gaussian", "Gaussian Noise"),
        ("salt_pepper", "Salt-Pepper Noise"),
    ]):
        ax = axes[ax_idx]
        results_dict = robustness_results_all.get(noise_name, {})
        if not results_dict:
            # Try the old format (flat dict of model->levels)
            results_dict = robustness_results_all

        markers = ["o", "s", "^", "D", "v"]
        for i, (model_name, results) in enumerate(results_dict.items()):
            levels = list(results.keys())
            accs = list(results.values())
            ax.plot(levels, accs, marker=markers[i % len(markers)],
                    label=model_name.upper(), linewidth=2, markersize=8)
        ax.set_xlabel("Noise Level (intensity)", fontsize=12)
        ax.set_ylabel("Test Accuracy", fontsize=12)
        ax.set_title(f"Robustness Under {noise_label}", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Model Robustness Comparison (Dual Noise Types)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_per_class_metrics(all_results, save_path=None):
    """Plot per-class F1 scores comparison across models."""
    if save_path is None:
        save_path = f"{FIG_DIR}/per_class_f1.png"

    first_model = list(all_results.values())[0]
    class_names = first_model.get("class_names", first_model.get("per_class", {}).keys())
    if not class_names:
        return None
    class_names = list(class_names)

    model_names = list(all_results.keys())
    x = np.arange(len(class_names))
    width = 0.2
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette("Set2", n_colors=n_models)

    for i, (model_name, res) in enumerate(all_results.items()):
        per_class = res.get("per_class", {})
        f1_vals = []
        for cls_name in class_names:
            if cls_name in per_class:
                f1_vals.append(per_class[cls_name]["f1"])
            else:
                # Try numeric index
                idx = class_names.index(cls_name)
                f1_vals.append(
                    per_class.get(list(per_class.keys())[idx] if per_class else cls_name, {})
                    .get("f1", 0) if isinstance(per_class, dict) else 0
                )
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_vals, width, label=model_name.upper(),
                      color=colors[i], edgecolor="white")
        for bar, v in zip(bars, f1_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", fontsize=7, rotation=90)

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.75, 1.05)
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
            ax.plot(range(1, len(val_loss) + 1), val_loss,
                    label="Val Loss", linewidth=2, linestyle="--")
        ax.set_title(f"{model_name.upper()} Loss Curve", fontweight="bold")
        ax.set_xlabel("Iteration / Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training & Validation Loss Curves (Iterative Models)", fontsize=14, fontweight="bold")
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

    # Format y-axis to better show differences
    min_val = min(min(train_accs.values()), min(test_accs.values())) - 0.02
    ax.set_ylim(min_val, 1.01)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_all_results(all_results, all_robustness, train_accs, test_accs, train_times=None):
    """Save all results to JSON."""
    output = {
        "metrics": {},
        "robustness": all_robustness,
        "overfitting": {"train_acc": train_accs, "test_acc": test_accs},
    }
    if train_times:
        output["training_times"] = train_times

    for model_name, res in all_results.items():
        output["metrics"][model_name] = {
            "accuracy": res["accuracy"],
            "precision_macro": res["precision_macro"],
            "recall_macro": res["recall_macro"],
            "f1_macro": res["f1_macro"],
            "precision_weighted": res["precision_weighted"],
            "recall_weighted": res["recall_weighted"],
            "f1_weighted": res["f1_weighted"],
            "inference_time_ms": res["inference_time_ms"],
            "per_class": res.get("per_class", {}),
            "confusion_matrix": res["confusion_matrix"],
        }

    path = f"{RESULT_DIR}/all_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to {path}")
