"""
Feature selection module for CICIoT2023.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    from .data_pipeline import EXPECTED_FEATURE_COLUMNS, PROCESSED_DATA_DIR, REMOVE_FEATURES
except ImportError:
    from data_pipeline import EXPECTED_FEATURE_COLUMNS, PROCESSED_DATA_DIR, REMOVE_FEATURES

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
FEATURE_FILE = PROCESSED_DATA_DIR / "selected_features.txt"


def perform_feature_selection() -> None:
    print("Loading train and test datasets...")
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError("Train or test CSV file missing.")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    selected_features = EXPECTED_FEATURE_COLUMNS

    print("\n=== Feature Selection ===")
    print(f"Removed features:  {len(REMOVE_FEATURES)} -> {REMOVE_FEATURES}")
    print(f"Selected features: {len(selected_features)}")

    train_selected = train[selected_features + ["Attack_Category"]]
    test_selected = test[selected_features + ["Attack_Category"]]

    train_output = PROCESSED_DATA_DIR / "train_selected.csv"
    test_output = PROCESSED_DATA_DIR / "test_selected.csv"

    train_selected.to_csv(train_output, index=False)
    test_selected.to_csv(test_output, index=False)

    with open(FEATURE_FILE, "w", encoding="utf-8") as f:
        for feature in selected_features:
            f.write(feature + "\n")

    print(f"\nSaved selected feature datasets:\n  - {train_output}\n  - {test_output}\n  - {FEATURE_FILE}")


if __name__ == "__main__":
    perform_feature_selection()
