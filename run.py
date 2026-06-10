#!/usr/bin/env python3
"""CLI entry point for the Dry Bean ML project.

Usage:
    python run.py --algorithm svm
    python run.py --algorithm all
    python run.py --algorithm svm --pca
    python run.py --preprocess-only
    python run.py --analysis-only
    python run.py --all
"""

import argparse
import sys
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FIG_DIR, RESULT_DIR
from data_loader import load_raw_data, load_processed_data, get_class_names
from preprocess import Preprocessor
from analysis import analyze_data
from train import train_model, get_available_models
from evaluate import (
    evaluate_model, evaluate_robustness,
    plot_confusion_matrix, plot_metrics_comparison,
    plot_robustness_curves, plot_loss_curves,
    plot_overfitting_analysis, save_all_results,
)


def run_analysis():
    """Run data analysis and save visualizations."""
    print("\n" + "=" * 60)
    print("  STEP 1: DATA ANALYSIS")
    print("=" * 60)
    analyze_data(save=True)


def run_preprocessing(use_pca=False):
    """Run preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("  STEP 2: DATA PREPROCESSING")
    print("=" * 60)
    preprocessor = Preprocessor(use_pca=use_pca)
    result = preprocessor.run()
    return result


def run_training(X_train, y_train, X_val, y_val, X_test, y_test, algorithms):
    """Train specified algorithms and return results."""
    print("\n" + "=" * 60)
    print("  STEP 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    class_names = get_class_names()

    all_results = {}
    all_robustness = {}
    train_accs = {}
    test_accs = {}
    histories = {}

    for algo in algorithms:
        print(f"\n{'─' * 50}")
        print(f"  Training: {algo.upper()}")
        print(f"{'─' * 50}")

        model, history, train_time, train_acc = train_model(
            algo, X_train, y_train, X_val, y_val
        )
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Training accuracy: {train_acc:.4f}")

        print(f"  Evaluating on test set...")
        metrics = evaluate_model(model, X_test, y_test, algo, class_names)
        all_results[algo] = metrics
        train_accs[algo] = train_acc
        test_accs[algo] = metrics["accuracy"]
        histories[algo] = history

        print(f"  Test accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Test F1 (macro):  {metrics['f1_macro']:.4f}")
        print(f"  Test F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Inference time:  {metrics['inference_time_ms']:.4f} ms/sample")

        # Robustness evaluation
        print(f"  Evaluating robustness...")
        rob = evaluate_robustness(model, X_test, y_test, algo, "gaussian")
        all_robustness[algo] = rob
        for level, acc in rob.items():
            print(f"    Noise level {level:.2f}: accuracy = {acc:.4f}")

        # Confusion matrix
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]), class_names, algo
        )
        print(f"  Confusion matrix saved: {FIG_DIR}/cm_{algo}.png")

    return all_results, all_robustness, train_accs, test_accs, histories


def run_evaluation_plots(all_results, all_robustness, train_accs, test_accs, histories):
    """Generate comparison plots."""
    print("\n" + "=" * 60)
    print("  STEP 4: GENERATING COMPARISON PLOTS")
    print("=" * 60)

    plot_metrics_comparison(all_results)
    print(f"  Metrics comparison saved: {FIG_DIR}/metrics_comparison.png")

    plot_robustness_curves(all_robustness)
    print(f"  Robustness curves saved: {FIG_DIR}/robustness_curves.png")

    plot_loss_curves(histories)
    print(f"  Loss curves saved: {FIG_DIR}/loss_curves.png")

    plot_overfitting_analysis(train_accs, test_accs)
    print(f"  Overfitting analysis saved: {FIG_DIR}/overfitting_analysis.png")

    save_all_results(all_results, all_robustness, train_accs, test_accs, histories)


def main():
    parser = argparse.ArgumentParser(
        description="Dry Bean Classification - ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --algorithm svm
  python run.py --algorithm all
  python run.py --algorithm svm,xgboost
  python run.py --pca
  python run.py --all
  python run.py --analysis-only
  python run.py --preprocess-only
        """,
    )
    parser.add_argument(
        "--algorithm", type=str, default="all",
        help=f"Algorithm(s) to use, comma-separated. Available: {get_available_models()}, all (default: all)"
    )
    parser.add_argument(
        "--pca", action="store_true",
        help="Use PCA dimensionality reduction"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run full pipeline: analysis + preprocessing + training + evaluation"
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Run only data analysis"
    )
    parser.add_argument(
        "--preprocess-only", action="store_true",
        help="Run only preprocessing"
    )
    args = parser.parse_args()

    if args.analysis_only:
        run_analysis()
        return

    if args.preprocess_only:
        run_preprocessing(use_pca=args.pca)
        return

    # Determine which algorithms to run
    if args.algorithm == "all":
        algorithms = get_available_models()
    else:
        algorithms = [a.strip() for a in args.algorithm.split(",")]
        for a in algorithms:
            if a not in get_available_models():
                print(f"Error: Unknown algorithm '{a}'. Available: {get_available_models()}")
                sys.exit(1)

    if args.all:
        run_analysis()

    # Preprocessing
    run_preprocessing(use_pca=args.pca)

    # Load processed data
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_processed_data()
    print(f"Loaded processed data: X_train={X_train.shape}, X_test={X_test.shape}")

    # Training & evaluation
    results = run_training(X_train, y_train, X_val, y_val, X_test, y_test, algorithms)
    all_results, all_robustness, train_accs, test_accs, histories = results

    # Comparison plots
    run_evaluation_plots(all_results, all_robustness, train_accs, test_accs, histories)

    # Print final summary table
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Wtd':>10} {'Time(ms)':>10} {'Train Acc':>10}")
    print("-" * 62)
    for algo in algorithms:
        m = all_results[algo]
        print(f"{algo.upper():<12} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
              f"{m['f1_weighted']:>10.4f} {m['inference_time_ms']:>10.4f} {train_accs[algo]:>10.4f}")

    print("\nDone. All figures saved to output/figures/, results to output/results/")


if __name__ == "__main__":
    main()
