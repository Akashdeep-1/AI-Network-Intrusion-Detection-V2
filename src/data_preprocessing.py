"""
CICIoT2023 Preprocessing Pipeline.

Processes raw CICIoT2023 CSV files into a deduplicated, balanced, clean dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .data_pipeline import (
        EXPECTED_FEATURE_COLUMNS,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        REMOVE_FEATURES,
        clean_and_deduplicate,
    )
    from .label_mapping import HIGH_LEVEL_CLASSES, LABEL_MAPPING
except ImportError:
    from data_pipeline import (
        EXPECTED_FEATURE_COLUMNS,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        REMOVE_FEATURES,
        clean_and_deduplicate,
    )
    from label_mapping import HIGH_LEVEL_CLASSES, LABEL_MAPPING

RANDOM_STATE = 42
CHUNK_SIZE = 100_000

TARGET_PER_CLASS = {
    "Benign": 100_000,
    "DDoS": 100_000,
    "DoS": 100_000,
    "Mirai": 100_000,
    "Recon": 75_000,
    "Spoofing": 75_000,
    "WebAttack": 50_000,
    "BruteForce": 25_000,
    "OtherAttack": 25_000,
}


def create_processed_dataset() -> pd.DataFrame:
    print("=" * 70)
    print("CICIoT2023 PREPROCESSING & DEDUPLICATION")
    print("=" * 70)

    files = sorted(RAW_DATA_DIR.glob("Merged*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DATA_DIR}")

    print(f"Found {len(files)} raw CSV files in {RAW_DATA_DIR}")

    chunks = []
    total_raw_rows = 0

    # Stream chunks, clean, map labels, drop invalid
    for index, file in enumerate(files, start=1):
        print(f"\rProcessing [{index}/{len(files)}] {file.name}...", end="", flush=True)
        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
            total_raw_rows += len(chunk)
            chunk.columns = [str(col).strip() for col in chunk.columns]
            if "Label" not in chunk.columns:
                continue
            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            chunk["Attack_Category"] = chunk["Label"].map(LABEL_MAPPING)
            chunk = chunk.dropna(subset=["Attack_Category"])

            feature_cols = [c for c in chunk.columns if c not in ("Label", "Attack_Category")]
            for col in feature_cols:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunk[feature_cols] = chunk[feature_cols].replace([np.inf, -np.inf], np.nan)
            chunk = chunk.dropna(subset=feature_cols)

            if not chunk.empty:
                chunks.append(chunk)

    print("\nCombining raw chunks...")
    full_df = pd.concat(chunks, ignore_index=True)

    # Global deduplication to prevent data leakage
    before_dedup = len(full_df)
    full_df = full_df.drop_duplicates(subset=EXPECTED_FEATURE_COLUMNS).reset_index(drop=True)
    duplicates_removed = before_dedup - len(full_df)

    print(f"Raw rows scanned:       {total_raw_rows:,}")
    print(f"Clean valid rows:       {before_dedup:,}")
    print(f"Duplicates removed:     {duplicates_removed:,}")
    print(f"Unique clean rows:      {len(full_df):,}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save ciciot2023_clean.csv (authoritative clean deduplicated dataset)
    clean_file = PROCESSED_DATA_DIR / "ciciot2023_clean.csv"
    full_df.to_csv(clean_file, index=False)

    # Perform global class-aware random sampling for ciciot2023_balanced.csv
    sampled_parts = []
    for class_name, target in TARGET_PER_CLASS.items():
        class_subset = full_df[full_df["Attack_Category"] == class_name]
        take = min(target, len(class_subset))
        if take > 0:
            sampled_parts.append(class_subset.sample(n=take, random_state=RANDOM_STATE))

    balanced_df = pd.concat(sampled_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    balanced_file = PROCESSED_DATA_DIR / "ciciot2023_balanced.csv"
    balanced_df.to_csv(balanced_file, index=False)

    # Save pipeline metadata
    meta = {
        "total_raw_rows": total_raw_rows,
        "clean_rows": len(full_df),
        "duplicates_removed": duplicates_removed,
        "balanced_rows": len(balanced_df),
        "clean_class_distribution": full_df["Attack_Category"].value_counts().to_dict(),
        "balanced_class_distribution": balanced_df["Attack_Category"].value_counts().to_dict(),
    }
    with open(PROCESSED_DATA_DIR / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:\n  - {clean_file}\n  - {balanced_file}\n  - {PROCESSED_DATA_DIR / 'preprocessing_metadata.json'}")
    return full_df


if __name__ == "__main__":
    create_processed_dataset()
