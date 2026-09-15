"""
Leakage-free dataset splitting for CICIoT2023.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

try:
    from .data_pipeline import (
        EXPECTED_FEATURE_COLUMNS,
        PROCESSED_DATA_DIR,
        clean_and_deduplicate,
        make_splits,
    )
except ImportError:
    from data_pipeline import (
        EXPECTED_FEATURE_COLUMNS,
        PROCESSED_DATA_DIR,
        clean_and_deduplicate,
        make_splits,
    )

INPUT_FILE = PROCESSED_DATA_DIR / "ciciot2023_clean.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def split_data() -> None:
    print("Loading cleaned dataset...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Clean dataset not found at {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"Initial rows: {len(df):,}")

    # Ensure global deduplication before splitting to prevent leakage
    df, clean_meta = clean_and_deduplicate(df)
    print(f"Rows after deduplication: {len(df):,}")
    print(f"Duplicates removed: {clean_meta['duplicate_feature_rows_removed']:,}")

    train_df, _, test_df, split_meta = make_splits(
        df, test_size=TEST_SIZE, validation_size=0.0, random_state=RANDOM_STATE
    )

    # Combine metadata
    meta = {**clean_meta, **split_meta}

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    with open(PROCESSED_DATA_DIR / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n========================================")
    print("LEAKAGE-FREE SPLIT COMPLETE")
    print("========================================")
    print(f"Train rows:               {len(train_df):,}")
    print(f"Test rows:                {len(test_df):,}")
    print(f"Train/test row overlap:   {meta['train_test_overlap_count']}")
    print(f"Saved files:\n  - {TRAIN_FILE}\n  - {TEST_FILE}\n  - {PROCESSED_DATA_DIR / 'split_metadata.json'}")


if __name__ == "__main__":
    split_data()
