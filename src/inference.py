"""
Shared inference logic for AI Network Intrusion Detection System.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from .data_pipeline import EXPECTED_FEATURE_COLUMNS
except ImportError:
    from data_pipeline import EXPECTED_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPUTATION_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "numeric_imputation_stats.json"


class InferenceError(Exception):
    """Raised when inference validation or execution fails."""


def load_imputation_stats() -> Dict[str, Dict[str, float]]:
    """Load training-derived median imputation statistics."""
    if IMPUTATION_STATS_PATH.exists():
        return json.loads(IMPUTATION_STATS_PATH.read_text())
    return {}


def sanitize_features(
    df_features: pd.DataFrame,
    imputation_stats: Dict[str, Dict[str, float]] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Coerce to numeric, replace inf/-inf with NaN, and impute using training medians.
    Returns the sanitized DataFrame and a dict of counts of sanitized values per column.
    """
    if imputation_stats is None:
        imputation_stats = load_imputation_stats()

    df = df_features.copy()
    sanitized_counts: Dict[str, int] = {}

    for col in df.columns:
        original_count = len(df)
        numeric = pd.to_numeric(df[col], errors="coerce")

        # Count inf/-inf before replacement
        inf_count = int(np.isinf(numeric).sum())
        nan_count = int(numeric.isna().sum())
        total_invalid = inf_count + nan_count

        if total_invalid > 0:
            sanitized_counts[col] = total_invalid

        # Replace inf/-inf with NaN
        numeric = numeric.replace([np.inf, -np.inf], np.nan)

        # Impute NaN values using training-derived median
        if col in imputation_stats and numeric.isna().any():
            median_val = imputation_stats[col].get("median", 0.0)
            numeric = numeric.fillna(median_val)

        df[col] = numeric

    return df, sanitized_counts


def validate_feature_frame(
    df: pd.DataFrame,
    expected_features: List[str] | None = None,
) -> pd.DataFrame:
    """
    Validate input dataframe for inference schema and data integrity.
    Returns validated features; extra columns are ignored.
    """
    if expected_features is None:
        expected_features = EXPECTED_FEATURE_COLUMNS

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Ignore target columns if user uploaded ground truth
    feature_cols = [c for c in df.columns if c not in ("Label", "Attack_Category")]

    missing = set(expected_features) - set(feature_cols)
    if missing:
        raise InferenceError(
            f"Input dataframe missing required feature columns: {sorted(list(missing))}"
        )

    # Reorder columns to exact expected feature ordering
    df_features = df[expected_features].copy()
    return df_features


def predict_dataframe(
    model_artifact: Dict[str, Any],
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run inference using a loaded model artifact.
    Returns predictions and a metadata dict including sanitization info.
    """
    import json
    from pathlib import Path

    model = model_artifact["model"]
    label_encoder = model_artifact["label_encoder"]
    feature_names = model_artifact.get("feature_names", EXPECTED_FEATURE_COLUMNS)
    scaler = model_artifact.get("scaler")

    # Structural validation
    df_features = validate_feature_frame(df, expected_features=feature_names)

    # Numeric sanitization using training-derived statistics
    imputation_stats = load_imputation_stats()
    df_features, sanitized_counts = sanitize_features(df_features, imputation_stats)

    # Scale if applicable
    if scaler is not None:
        X = scaler.transform(df_features)
    else:
        X = df_features

    # Predict
    raw_preds = model.predict(X)
    pred_indices = np.asarray(raw_preds).astype(int)
    pred_labels = label_encoder.inverse_transform(pred_indices)

    # Confidence scores
    confidences = np.ones(len(df_features), dtype=float)
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            confidences = np.max(probs, axis=1)
        except Exception:
            pass

    # Build results
    is_attack = [label != "Benign" for label in pred_labels]
    status = ["ALERT: Malicious Traffic" if attack else "NORMAL: Benign Traffic" for attack in is_attack]

    results_df = pd.DataFrame(
        {
            "Row_ID": np.arange(1, len(df_features) + 1),
            "Predicted_Category": pred_labels,
            "Confidence": np.round(confidences * 100, 2),
            "Is_Attack": is_attack,
            "Attack_Status": status,
        }
    )

    metadata = {
        "total_rows": len(df_features),
        "sanitized_values": sanitized_counts,
        "model_name": model_artifact.get("model_name", "Unknown"),
    }

    return results_df, metadata
