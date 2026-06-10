"""Data loading module for the Dry Bean Dataset."""

import pandas as pd
import numpy as np
from config import TRAIN_PATH, TEST_PATH, VAL_PATH, FEATURE_COLS, TARGET_COL


def load_raw_data():
    """Load raw CSV files. Returns (train_df, test_df, val_df)."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # Fix BOM in column names
    for df in [train_df, test_df, val_df]:
        df.columns = [c.lstrip("﻿") for c in df.columns]

    return train_df, test_df, val_df


def load_clean_data():
    """Load cleaned CSV files from output directory."""
    from config import CLEAN_TRAIN_PATH, CLEAN_TEST_PATH, CLEAN_VAL_PATH

    train_df = pd.read_csv(CLEAN_TRAIN_PATH)
    test_df = pd.read_csv(CLEAN_TEST_PATH)
    val_df = pd.read_csv(CLEAN_VAL_PATH)
    return train_df, test_df, val_df


def load_processed_data():
    """Load preprocessed NumPy arrays."""
    from config import PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, PROCESSED_VAL_PATH

    train = np.load(PROCESSED_TRAIN_PATH)
    test = np.load(PROCESSED_TEST_PATH)
    val = np.load(PROCESSED_VAL_PATH)
    return (train["X"], train["y"]), (test["X"], test["y"]), (val["X"], val["y"])


EXPECTED_CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]


def get_class_names(y=None):
    """Return sorted class names (canonical 7 classes)."""
    return list(EXPECTED_CLASSES)
