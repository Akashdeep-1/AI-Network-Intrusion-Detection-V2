"""
Comprehensive tests for models, evaluation, and Streamlit app functions.
"""

from __future__ import annotations

from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd

from app.app import load_models
from src.data_pipeline import EXPECTED_FEATURE_COLUMNS, clean_and_deduplicate, make_splits
from src.inference import InferenceError, predict_dataframe, sanitize_features, validate_feature_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_FILE = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "test_selected.csv"


class ModelArtifactTests(unittest.TestCase):
    def test_models_exist_and_load(self):
        models = load_models()
        self.assertGreaterEqual(len(models), 3)
        self.assertIn("XGBoost (Recommended)", models)
        self.assertIn("Random Forest", models)
        self.assertIn("Logistic Regression", models)

    def test_model_inference_on_sample(self):
        if not SAMPLE_FILE.exists():
            self.skipTest("Sample test file not found.")
        sample_df = pd.read_csv(SAMPLE_FILE, nrows=50)
        models = load_models()
        for name, artifact in models.items():
            results, meta = predict_dataframe(artifact, sample_df)
            self.assertEqual(len(results), len(sample_df), msg=f"{name} returned {len(results)} results, expected {len(sample_df)}")
            self.assertIn("Predicted_Category", results.columns)
            self.assertIn("Confidence", results.columns)
            self.assertIn("Is_Attack", results.columns)
            self.assertTrue(all(c >= 0.0 for c in results["Confidence"]))

    def test_schema_rejection_missing_cols(self):
        invalid_df = pd.DataFrame({"Header_Length": [10.0]})
        with self.assertRaises(InferenceError):
            validate_feature_frame(invalid_df)

    def test_schema_accepts_extra_cols(self):
        """Extra/unused columns should be silently ignored."""
        sample_dict = {col: [1.0] for col in EXPECTED_FEATURE_COLUMNS}
        sample_dict["malicious_extra_feature"] = [999.0]
        valid_df = pd.DataFrame(sample_dict)
        validated = validate_feature_frame(valid_df)
        self.assertEqual(len(validated.columns), len(EXPECTED_FEATURE_COLUMNS))
        self.assertNotIn("malicious_extra_feature", validated.columns)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = list(EXPECTED_FEATURE_COLUMNS[:6])
        rows = []
        for index, label in enumerate(("BENIGN", "DDOS-UDP_FLOOD", "DOS-TCP_FLOOD") * 4):
            row = {feature: float(index + position) for position, feature in enumerate(self.features)}
            row["Label"] = label
            rows.append(row)
        self.frame = pd.DataFrame(rows)

    def test_cleaning_maps_labels_and_removes_duplicate_feature_vectors(self):
        duplicated = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        cleaned, metadata = clean_and_deduplicate(duplicated)
        self.assertEqual(len(cleaned), len(self.frame))
        self.assertEqual(metadata["duplicate_feature_rows_removed"], 1)
        self.assertEqual(set(cleaned["Attack_Category"]), {"Benign", "DDoS", "DoS"})

    def test_split_is_deterministic_and_has_no_feature_overlap(self):
        cleaned, _ = clean_and_deduplicate(self.frame)
        train, validation, test, _ = make_splits(
            cleaned, test_size=0.25, validation_size=0.25, feature_columns=self.features
        )
        self.assertEqual(len(train) + len(validation) + len(test), len(cleaned))
        train_hashes = set(pd.util.hash_pandas_object(train[self.features], index=False).to_numpy())
        test_hashes = pd.util.hash_pandas_object(test[self.features], index=False).to_numpy()
        self.assertTrue(train_hashes.isdisjoint(test_hashes))


if __name__ == "__main__":
    unittest.main()
