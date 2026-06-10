"""Configuration and path definitions for the Dry Bean ML project."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DryBeanDataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

TRAIN_PATH = os.path.join(DATA_DIR, "Dry_Bean_Dataset_Dirty_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "Dry_Bean_Dataset_Dirty_test.csv")
VAL_PATH = os.path.join(DATA_DIR, "Dry_Bean_Dataset_Dirty_val.csv")

CLEAN_TRAIN_PATH = os.path.join(OUTPUT_DIR, "train_clean.csv")
CLEAN_TEST_PATH = os.path.join(OUTPUT_DIR, "test_clean.csv")
CLEAN_VAL_PATH = os.path.join(OUTPUT_DIR, "val_clean.csv")

PROCESSED_TRAIN_PATH = os.path.join(OUTPUT_DIR, "train_processed.npz")
PROCESSED_TEST_PATH = os.path.join(OUTPUT_DIR, "test_processed.npz")
PROCESSED_VAL_PATH = os.path.join(OUTPUT_DIR, "val_processed.npz")

for d in [OUTPUT_DIR, FIG_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 42
FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]
TARGET_COL = "Class"

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
