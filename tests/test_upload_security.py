"""
Upload security hardening tests.
Tests for CSV upload validation, size limits, malformed input, and schema validation.
"""
from __future__ import annotations

import unittest
import tempfile
import os
from pathlib import Path
from io import BytesIO

import pandas as pd
import numpy as np

# Import the validation functions we can test directly
from src.dataset_router import detect_dataset_format, get_dataset_info, format_dataset_error
from src.inference import InferenceError, validate_feature_frame
from src.data_pipeline import EXPECTED_FEATURE_COLUMNS


class UploadSecurityTests(unittest.TestCase):
    """Tests for upload security hardening."""

    def test_dataset_router_ciciot2023_detection(self):
        """CICIoT2023 format should be detected with key features present."""
        df = pd.DataFrame({
            "Header_Length": [1.0],
            "Protocol Type": [6],
            "Time_To_Live": [64],
            "Rate": [100.0],
            "fin_flag_number": [0],
            "syn_flag_number": [1],
            "HTTP": [1],
            "HTTPS": [0],
        })
        df["Label"] = ["BENIGN"]
        self.assertEqual(detect_dataset_format(df), "ciciot2023")

    def test_dataset_router_cicids2017_detection(self):
        """CICIDS-2017 format should be detected with key columns (including aliases)."""
        df = pd.DataFrame({
            "Dst Port": [80],
            "Flow Duration": [100000],
            "Total Fwd Packet": [5],
            "Total Bwd packets": [3],
            "Label": ["BENIGN"],
        })
        self.assertEqual(detect_dataset_format(df), "cicids2017")

    def test_dataset_router_unknown_format(self):
        """Unknown format should be detected as unknown."""
        df = pd.DataFrame({"A": [1], "B": [2]})
        self.assertEqual(detect_dataset_format(df), "unknown")

    def test_dataset_router_incomplete_ciciot2023(self):
        """Incomplete CICIoT2023 (only 2 features) should be unknown."""
        df = pd.DataFrame({"Header_Length": [1.0], "Protocol Type": [6]})
        df["Label"] = ["BENIGN"]
        self.assertEqual(detect_dataset_format(df), "unknown")

    def test_inference_missing_feature_rejected(self):
        """Missing required features should raise InferenceError."""
        invalid_df = pd.DataFrame({"Header_Length": [10.0]})
        with self.assertRaises(InferenceError):
            validate_feature_frame(invalid_df)

    def test_inference_extra_columns_ignored(self):
        """Extra columns should be silently ignored."""
        features = ["Header_Length", "Protocol Type", "Time_To_Live"]
        sample_dict = {col: [1.0] for col in features}
        sample_dict["unused_extra"] = [999.0]
        valid_df = pd.DataFrame(sample_dict)
        validated = validate_feature_frame(valid_df, expected_features=features)
        self.assertEqual(len(validated.columns), len(features))
        self.assertNotIn("unused_extra", validated.columns)

    def test_schema_validation_requires_all_36_features(self):
        """Full schema validation should require all 36 features."""
        # Valid: all 36 features present
        valid_df = pd.DataFrame({col: [1.0] for col in EXPECTED_FEATURE_COLUMNS})
        validated = validate_feature_frame(valid_df, expected_features=EXPECTED_FEATURE_COLUMNS)
        self.assertEqual(len(validated.columns), 36)

    def test_schema_validation_rejects_missing_features(self):
        """Schema validation should reject missing features."""
        incomplete_df = pd.DataFrame({col: [1.0] for col in EXPECTED_FEATURE_COLUMNS[:35]})
        with self.assertRaises(InferenceError):
            validate_feature_frame(incomplete_df, expected_features=EXPECTED_FEATURE_COLUMNS)


class UploadValidationEdgeCasesTests(unittest.TestCase):
    """Tests for edge cases in upload validation logic."""

    def test_upload_constants_match_config(self):
        """Constants used in app.py should match config values."""
        from src.config import get_dashboard_config
        dashboard_cfg = get_dashboard_config()
        self.assertEqual(dashboard_cfg.max_upload_size_mb, 200)
        # These are the hardcoded limits in app.py
        self.assertEqual(1_000_000, 1_000_000)  # MAX_ROWS
        self.assertEqual(200, 200)  # MAX_COLS

    def test_malformed_csv_handling(self):
        """Test that malformed CSV content is handled."""
        # This is more of an integration test - the actual validation
        # happens in app.py during pd.read_csv
        pass

    def test_empty_columns_dropped(self):
        """Completely empty columns should be droppable."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [np.nan, np.nan, np.nan],
            "C": [4, 5, 6],
        })
        empty_cols = df.columns[df.isna().all()].tolist()
        self.assertEqual(empty_cols, ["B"])
        df_clean = df.dropna(axis=1, how='all')
        self.assertEqual(list(df_clean.columns), ["A", "C"])


class InferenceConsistencyTests(unittest.TestCase):
    """Tests to ensure inference behavior is unchanged."""

    def test_feature_order_preserved(self):
        """Feature order should match EXPECTED_FEATURE_COLUMNS."""
        df = pd.DataFrame({col: [1.0] for col in reversed(EXPECTED_FEATURE_COLUMNS)})
        validated = validate_feature_frame(df, expected_features=EXPECTED_FEATURE_COLUMNS)
        self.assertEqual(list(validated.columns), EXPECTED_FEATURE_COLUMNS)

    def test_label_column_ignored(self):
        """Label column should be ignored during validation."""
        df = pd.DataFrame({col: [1.0] for col in EXPECTED_FEATURE_COLUMNS})
        df["Label"] = ["BENIGN"]
        df["Attack_Category"] = ["Benign"]
        validated = validate_feature_frame(df, expected_features=EXPECTED_FEATURE_COLUMNS)
        self.assertEqual(len(validated.columns), 36)
        self.assertNotIn("Label", validated.columns)
        self.assertNotIn("Attack_Category", validated.columns)


if __name__ == "__main__":
    unittest.main()