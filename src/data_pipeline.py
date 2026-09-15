"""
Authoritative, reproducible CICIoT2023 Data Pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .label_mapping import HIGH_LEVEL_CLASSES, LABEL_MAPPING
except ImportError:
    from label_mapping import HIGH_LEVEL_CLASSES, LABEL_MAPPING

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ciciot2023" / "MERGED_CSV"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ciciot2023"

REMOVE_FEATURES = ["IPv", "LLC", "Tot size"]

EXPECTED_FEATURE_COLUMNS = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "IAT",
    "Number",
    "Variance",
]

ALL_RAW_FEATURES = EXPECTED_FEATURE_COLUMNS + REMOVE_FEATURES


class DatasetValidationError(Exception):
    """Raised when dataset fails schema or data validation."""


def clean_and_deduplicate(
    df: pd.DataFrame,
    features_to_use: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and deduplicate input dataframe.
    """
    if features_to_use is None:
        features_to_use = EXPECTED_FEATURE_COLUMNS

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Label" not in df.columns and "Attack_Category" not in df.columns:
        raise DatasetValidationError("Dataset must contain 'Label' or 'Attack_Category' column.")

    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
        df["Attack_Category"] = df["Label"].map(LABEL_MAPPING)
        missing_mapped = df[df["Attack_Category"].isna()]
        if len(missing_mapped) > 0:
            unknown_labels = missing_mapped["Label"].unique().tolist()
            raise DatasetValidationError(f"Unknown labels encountered: {unknown_labels}")

    present_features = [col for col in features_to_use if col in df.columns]
    if not present_features:
        raise DatasetValidationError("None of the expected feature columns are present.")

    for col in present_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[present_features] = df[present_features].replace([np.inf, -np.inf], np.nan)

    initial_len = len(df)
    df = df.dropna(subset=present_features + ["Attack_Category"]).reset_index(drop=True)
    nan_removed = initial_len - len(df)

    # Deduplicate exact feature vectors to prevent train/test leakage
    before_dedup = len(df)
    df = df.drop_duplicates(subset=present_features).reset_index(drop=True)
    duplicates_removed = before_dedup - len(df)

    metadata = {
        "initial_rows": initial_len,
        "invalid_numeric_rows_removed": nan_removed,
        "duplicate_feature_rows_removed": duplicates_removed,
        "final_rows": len(df),
        "class_distribution": df["Attack_Category"].value_counts().to_dict(),
    }

    return df, metadata


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.0,
    random_state: int = 42,
    feature_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Create leakage-free, stratified train/val/test splits.
    """
    if feature_columns is None:
        feature_columns = EXPECTED_FEATURE_COLUMNS

    X = df[feature_columns]
    y = df["Attack_Category"]

    if validation_size > 0.0:
        val_test_ratio = test_size + validation_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=val_test_ratio, random_state=random_state, stratify=y
        )
        test_prop = test_size / val_test_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_prop, random_state=random_state, stratify=y_temp
        )
        val_df = X_val.copy()
        val_df["Attack_Category"] = y_val.values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        val_df = pd.DataFrame()

    train_df = X_train.copy()
    train_df["Attack_Category"] = y_train.values

    test_df = X_test.copy()
    test_df["Attack_Category"] = y_test.values

    # Verify no exact feature vector overlap
    train_hashes = set(pd.util.hash_pandas_object(train_df[feature_columns], index=False).to_numpy())
    test_hashes = pd.util.hash_pandas_object(test_df[feature_columns], index=False).to_numpy()
    test_overlap = sum(h in train_hashes for h in test_hashes)

    metadata = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_test_overlap_count": test_overlap,
        "train_class_distribution": train_df["Attack_Category"].value_counts().to_dict(),
        "test_class_distribution": test_df["Attack_Category"].value_counts().to_dict(),
    }

    return train_df, val_df, test_df, metadata
