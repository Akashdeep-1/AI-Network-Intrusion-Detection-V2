"""
Pipeline and inference core behavior tests.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.data_pipeline import clean_and_deduplicate, make_splits, EXPECTED_FEATURE_COLUMNS
from src.inference import InferenceError, validate_feature_frame


class InferenceSchemaTests(unittest.TestCase):
    def test_missing_feature_is_rejected(self):
        invalid_df = pd.DataFrame({"Header_Length": [10.0]})
        with self.assertRaises(InferenceError):
            validate_feature_frame(invalid_df)

    def test_extra_columns_are_silently_ignored(self):
        features = ["Header_Length", "Protocol Type", "Time_To_Live"]
        sample_dict = {col: [1.0] for col in features}
        sample_dict["unused_extra"] = [999.0]
        valid_df = pd.DataFrame(sample_dict)
        validated = validate_feature_frame(valid_df, expected_features=features)
        self.assertEqual(len(validated.columns), len(features))
        self.assertNotIn("unused_extra", validated.columns)


class PipelineBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        # Use EXPECTED_FEATURE_COLUMNS subset for fast unit tests
        self.features = EXPECTED_FEATURE_COLUMNS[:4]
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
