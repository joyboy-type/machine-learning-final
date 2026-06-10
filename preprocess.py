"""Data preprocessing: cleaning dirty labels, missing markers, feature engineering."""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import (
    FEATURE_COLS, TARGET_COL, RANDOM_SEED,
    CLEAN_TRAIN_PATH, CLEAN_TEST_PATH, CLEAN_VAL_PATH,
    PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, PROCESSED_VAL_PATH,
)
from data_loader import load_raw_data


# Mapping for known label corruptions
LABEL_CORRECTIONS = {
    "D3RMAS0N": "DERMASON",
    "H0R0Z": "HOROZ",
    "S3K3R": "SEKER",
    "B0MBAY": "BOMBAY",
}

EXPECTED_CLASSES = {"DERMASON", "SIRA", "SEKER", "HOROZ", "CALI", "BARBUNYA", "BOMBAY"}


def _standardize_class_label(label):
    """Map a possibly corrupted class label to its canonical form."""
    label = str(label).strip().upper()
    if label in LABEL_CORRECTIONS:
        label = LABEL_CORRECTIONS[label]
    return label


def _clean_features(df):
    """Clean feature columns in-place."""
    # Solidity: replace "?" with NaN, then convert to float
    df["Solidity"] = df["Solidity"].replace("?", np.nan)
    df["Solidity"] = pd.to_numeric(df["Solidity"], errors="coerce")

    # Compactness: strip " cm" suffix, then convert to float
    df["Compactness"] = df["Compactness"].astype(str).str.replace(r"\s*cm", "", regex=True)
    df["Compactness"] = pd.to_numeric(df["Compactness"], errors="coerce")

    # Ensure all feature columns are numeric
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


class Preprocessor:
    """Pipeline for cleaning and transforming the Dry Bean dataset."""

    def __init__(self, use_pca=False, n_components=0.95):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=RANDOM_SEED) if use_pca else None
        self.use_pca = use_pca
        self.label_mapping = {}
        self.inverse_label_mapping = {}
        self.cleaning_stats = {}

    def clean(self, train_df, test_df, val_df):
        """Clean data: fix labels, missing values, remove bad rows."""
        stats = {}
        total_before = len(train_df) + len(test_df) + len(val_df)

        dfs_out = {}
        for name, df in [("train", train_df), ("test", test_df), ("val", val_df)]:
            before = len(df)
            stats[f"{name}_rows_raw"] = before

            # 1. Standardize class labels
            df[TARGET_COL] = df[TARGET_COL].apply(_standardize_class_label)

            # Remove rows with class labels that don't map to expected classes
            valid_mask = df[TARGET_COL].isin(EXPECTED_CLASSES)
            bad_labels = (~valid_mask).sum()
            stats[f"{name}_bad_labels_removed"] = bad_labels
            df = df[valid_mask].copy()

            # 2. Clean feature columns (fix "?", " cm", etc.)
            df = _clean_features(df)

            # Count missing before dropping
            missing_before = df[FEATURE_COLS].isnull().sum().sum()
            stats[f"{name}_missing_cells"] = missing_before

            # 3. Drop rows with any missing values
            rows_before_drop = len(df)
            df = df.dropna(subset=FEATURE_COLS)
            rows_dropped = rows_before_drop - len(df)
            stats[f"{name}_rows_dropped_missing"] = rows_dropped

            # 4. Remove duplicate rows
            dup_count = df.duplicated().sum()
            df = df.drop_duplicates()
            stats[f"{name}_duplicates_removed"] = dup_count

            # 5. Clip outliers at 3*IQR
            outlier_counts = {}
            for col in FEATURE_COLS:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 3.0 * IQR
                upper = Q3 + 3.0 * IQR
                n_out = ((df[col] < lower) | (df[col] > upper)).sum()
                if n_out > 0:
                    outlier_counts[col] = n_out
                df[col] = df[col].clip(lower, upper)
            stats[f"{name}_outliers_clipped"] = sum(outlier_counts.values())

            after = len(df)
            stats[f"{name}_rows_after"] = after
            stats[f"{name}_rows_removed_total"] = before - after
            dfs_out[name] = df

        total_after = sum(len(dfs_out[n]) for n in ["train", "test", "val"])
        stats["total_before"] = total_before
        stats["total_after"] = total_after
        stats["total_removed"] = total_before - total_after
        self.cleaning_stats = stats

        return dfs_out["train"], dfs_out["test"], dfs_out["val"]

    def fit_transform(self, train_df, test_df, val_df):
        """Fit scaler (+ PCA) on training data, transform all splits."""
        X_train = train_df[FEATURE_COLS].values.astype(np.float64)
        X_test = test_df[FEATURE_COLS].values.astype(np.float64)
        X_val = val_df[FEATURE_COLS].values.astype(np.float64)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_val = self.scaler.transform(X_val)

        if self.pca is not None:
            X_train = self.pca.fit_transform(X_train)
            X_test = self.pca.transform(X_test)
            X_val = self.pca.transform(X_val)

        class_names = sorted(EXPECTED_CLASSES)
        self.label_mapping = {name: i for i, name in enumerate(class_names)}
        self.inverse_label_mapping = {i: name for i, name in enumerate(class_names)}

        y_train = np.array([self.label_mapping[c] for c in train_df[TARGET_COL]])
        y_test = np.array([self.label_mapping[c] for c in test_df[TARGET_COL]])
        y_val = np.array([self.label_mapping[c] for c in val_df[TARGET_COL]])

        return (X_train, y_train), (X_test, y_test), (X_val, y_val)

    def save_processed(self, train_data, test_data, val_data):
        X_train, y_train = train_data
        X_test, y_test = test_data
        X_val, y_val = val_data
        np.savez_compressed(PROCESSED_TRAIN_PATH, X=X_train, y=y_train)
        np.savez_compressed(PROCESSED_TEST_PATH, X=X_test, y=y_test)
        np.savez_compressed(PROCESSED_VAL_PATH, X=X_val, y=y_val)

    def save_clean_csv(self, train_df, test_df, val_df):
        train_df.to_csv(CLEAN_TRAIN_PATH, index=False)
        test_df.to_csv(CLEAN_TEST_PATH, index=False)
        val_df.to_csv(CLEAN_VAL_PATH, index=False)

    def run(self):
        """Run full preprocessing pipeline."""
        print("=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)

        train_df, test_df, val_df = load_raw_data()
        print(f"\nRaw data loaded: train={len(train_df)}, test={len(test_df)}, val={len(val_df)}")

        train_df, test_df, val_df = self.clean(train_df, test_df, val_df)
        print("\n--- Cleaning Report ---")
        for k, v in self.cleaning_stats.items():
            print(f"  {k}: {v}")

        self.save_clean_csv(train_df, test_df, val_df)

        (X_train, y_train), (X_test, y_test), (X_val, y_val) = self.fit_transform(
            train_df, test_df, val_df
        )
        print(f"\n--- After Transformation ---")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape:  {X_test.shape}")
        print(f"  X_val shape:   {X_val.shape}")
        if self.pca:
            cumsum = self.pca.explained_variance_ratio_.cumsum()[-1]
            print(f"  PCA explained variance ratio (cumsum): {cumsum:.4f}")
            print(f"  PCA n_components: {self.pca.n_components_}")

        self.save_processed(train_data=(X_train, y_train),
                            test_data=(X_test, y_test),
                            val_data=(X_val, y_val))
        print("\nProcessed data saved. Preprocessing complete.\n")
        return (X_train, y_train), (X_test, y_test), (X_val, y_val), self


if __name__ == "__main__":
    preprocessor = Preprocessor(use_pca=False)
    preprocessor.run()
