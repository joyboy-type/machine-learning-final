"""Model training module: SVM, RandomForest, XGBoost, MLP classifiers."""

import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from config import RANDOM_SEED


MODEL_REGISTRY = {}


def register(name):
    """Decorator to register a model builder function."""
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


@register("svm")
def build_svm():
    return SVC(
        kernel="rbf", C=10.0, gamma="scale",
        probability=True, random_state=RANDOM_SEED,
    ), False  # is_iterative = False


@register("rf")
def build_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=RANDOM_SEED, n_jobs=-1,
    ), False


@register("xgboost")
def build_xgboost():
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1,
        use_label_encoder=False,
    ), True  # is_iterative = True (has loss curve from evals_result)


@register("mlp")
def build_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu",
        solver="adam", alpha=0.0001, batch_size=64,
        learning_rate="adaptive", max_iter=300,
        early_stopping=True, validation_fraction=0.1,
        random_state=RANDOM_SEED, verbose=False,
    ), True  # is_iterative = True (has loss_curve_ attribute)


def get_model(model_name):
    """Retrieve a model builder by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]()


def train_model(model_name, X_train, y_train, X_val=None, y_val=None):
    """Train a model and return (model, train_history).
    train_history = None for non-iterative models, or dict with 'train_loss', 'val_loss'.
    """
    model, is_iterative = get_model(model_name)
    history = None

    t0 = time.perf_counter()

    if model_name == "xgboost":
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, verbose=False)
        evals = model.evals_result()
        history = {
            "train_loss": evals["validation_0"]["mlogloss"],
            "val_loss": evals["validation_1"]["mlogloss"] if "validation_1" in evals else None,
        }
    elif model_name == "mlp":
        model.fit(X_train, y_train)
        history = {
            "train_loss": model.loss_curve_,
            "val_loss": model.validation_scores_ if hasattr(model, "validation_scores_") else None,
        }
    else:
        model.fit(X_train, y_train)

    train_time = time.perf_counter() - t0

    # Compute training accuracy for overfitting analysis
    train_acc = model.score(X_train, y_train)

    return model, history, train_time, train_acc


def get_available_models():
    """Return list of registered model names."""
    return list(MODEL_REGISTRY.keys())
