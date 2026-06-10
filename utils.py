"""Utility functions: noise generation, timing, result formatting."""

import time
import numpy as np
from config import RANDOM_SEED


def add_gaussian_noise(X, level):
    """Add Gaussian noise scaled by feature std * level."""
    rng = np.random.RandomState(RANDOM_SEED)
    noise = rng.randn(*X.shape) * X.std(axis=0) * level
    return X + noise


def add_salt_pepper_noise(X, level):
    """Add salt-and-pepper noise: randomly replace 'level' fraction of values
    with min or max of that feature."""
    rng = np.random.RandomState(RANDOM_SEED)
    X_noisy = X.copy()
    n_samples, n_features = X.shape
    n_corrupt = int(n_samples * n_features * level)
    for _ in range(n_corrupt):
        i = rng.randint(0, n_samples)
        j = rng.randint(0, n_features)
        if rng.rand() < 0.5:
            X_noisy[i, j] = X[:, j].min()
        else:
            X_noisy[i, j] = X[:, j].max()
    return X_noisy


def time_inference(model, X):
    """Measure average per-sample inference time in milliseconds."""
    n_samples = X.shape[0]
    # Warm-up
    model.predict(X[:min(10, n_samples)])
    start = time.perf_counter()
    model.predict(X)
    elapsed = time.perf_counter() - start
    return (elapsed / n_samples) * 1000  # ms per sample


def format_results_table(results_dict):
    """Format a dictionary of results as a markdown table string."""
    if not results_dict:
        return ""
    headers = list(next(iter(results_dict.values())).keys())
    table = "| Algorithm | " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["---"] * (len(headers) + 1)) + "|\n"
    for algo, metrics in results_dict.items():
        vals = []
        for h in headers:
            v = metrics.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        table += f"| {algo} | " + " | ".join(vals) + " |\n"
    return table
