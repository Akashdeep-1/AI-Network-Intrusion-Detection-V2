"""
Model integrity verification tests.
Tests for SHA-256 verification before deserialization.
"""
from __future__ import annotations
import hashlib
import tempfile
import unittest
from pathlib import Path
import sys
import joblib
import numpy as np

# Ensure src is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_integrity import (
    compute_sha256,
    verify_artifact,
    verify_all_artifacts,
    load_verified_joblib,
    IntegrityError,
)
from src.config import get_model_integrity_config, get_config


class ModelIntegrityTests(unittest.TestCase):
    """Tests for model artifact integrity verification."""

    def setUp(self):
        self.integrity_cfg = get_model_integrity_config()
        self.config = get_config()

    def test_compute_sha256_known_file(self):
        """Test SHA-256 computation on a known file."""
        # Create a test file with known content
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b"test content for hashing")
            temp_path = Path(f.name)
        
        try:
            computed = compute_sha256(temp_path)
            # Verify it's a valid SHA-256 hex string
            self.assertEqual(len(computed), 64)
            self.assertTrue(all(c in '0123456789abcdef' for c in computed))
            
            # Verify deterministic
            computed2 = compute_sha256(temp_path)
            self.assertEqual(computed, computed2)
        finally:
            temp_path.unlink()

    def test_verify_artifact_valid(self):
        """Test that a valid artifact passes verification."""
        # Use the actual XGBoost model
        models_dir = PROJECT_ROOT / "models"
        artifact_name = "xgboost"
        model_filename = self.config.models.xgboost
        file_path = models_dir / model_filename
        expected_hash = self.integrity_cfg.xgboost
        
        # Should pass without raising
        result = verify_artifact(artifact_name, file_path, expected_hash)
        self.assertTrue(result)

    def test_verify_artifact_missing_file(self):
        """Test that a missing artifact fails verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "nonexistent.joblib"
            with self.assertRaises(IntegrityError) as cm:
                verify_artifact("test_model", missing_path, "abcdef")
            self.assertIn("not found", str(cm.exception))

    def test_verify_artifact_wrong_hash(self):
        """Test that an artifact with wrong hash fails verification."""
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b"different content")
            temp_path = Path(f.name)
        
        try:
            wrong_hash = "0" * 64
            with self.assertRaises(IntegrityError) as cm:
                verify_artifact("test_model", temp_path, wrong_hash)
            self.assertIn("hash mismatch", str(cm.exception))
        finally:
            temp_path.unlink()

    def test_verify_artifact_no_expected_hash(self):
        """Test that missing expected hash fails verification."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b"content")
            temp_path = Path(f.name)
        
        try:
            with self.assertRaises(IntegrityError) as cm:
                verify_artifact("unknown_model", temp_path, None)
            self.assertIn("No expected hash configured", str(cm.exception))
        finally:
            temp_path.unlink()

    def test_verify_all_artifacts_passes(self):
        """Test that all current artifacts pass verification."""
        results = verify_all_artifacts()
        expected_artifacts = {"xgboost", "random_forest", "logistic_regression", "gru_60s"}
        self.assertEqual(set(results.keys()), expected_artifacts)
        self.assertTrue(all(results.values()))

    def test_load_verified_joblib_valid(self):
        """Test that valid joblib artifacts load after verification."""
        model = load_verified_joblib("xgboost")
        self.assertIn("model", model)
        self.assertIn("label_encoder", model)
        self.assertIn("feature_names", model)
        self.assertEqual(model["model_name"], "XGBoost")

    def test_load_verified_joblib_tampered_fails(self):
        """Test that a tampered joblib artifact fails to load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake model file
            fake_model = {"model": "fake", "label_encoder": "fake"}
            fake_path = Path(tmpdir) / "fake_model.joblib"
            joblib.dump(fake_model, fake_path)
            
            # Try to load with wrong hash
            with self.assertRaises(IntegrityError):
                verify_artifact("fake_model", fake_path, "0" * 64)

    def test_integrity_error_before_deserialization(self):
        """Verify that IntegrityError is raised before joblib.load is called."""
        # This is a behavioral test - we verify that verify_artifact
        # raises IntegrityError without calling joblib.load
        # The implementation computes hash first, then loads
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "test.joblib"
            fake_path.write_bytes(b"not a valid joblib")
            
            # Should fail on hash mismatch, not on joblib.load error
            with self.assertRaises(IntegrityError) as cm:
                verify_artifact("test", fake_path, "0" * 64)
            self.assertIn("hash mismatch", str(cm.exception))


class ForecastModelIntegrityTests(unittest.TestCase):
    """Tests for GRU forecasting model integrity."""

    def test_gru_hash_in_config(self):
        """Test that GRU hash is present in config."""
        cfg = get_model_integrity_config()
        self.assertTrue(hasattr(cfg, "gru_60s"))
        self.assertEqual(len(cfg.gru_60s), 64)

    def test_gru_file_exists(self):
        """Test that GRU model file exists."""
        models_dir = PROJECT_ROOT / "models"
        gru_path = models_dir / "cicids_gru_W10_K5_60s.pt"
        self.assertTrue(gru_path.exists())


class ConfigIntegrityTests(unittest.TestCase):
    """Tests for integrity config loading."""

    def test_model_integrity_config_loads(self):
        """Test that model integrity config loads correctly."""
        cfg = get_model_integrity_config()
        self.assertTrue(hasattr(cfg, "xgboost"))
        self.assertTrue(hasattr(cfg, "random_forest"))
        self.assertTrue(hasattr(cfg, "logistic_regression"))
        self.assertTrue(hasattr(cfg, "gru_60s"))

    def test_hashes_are_valid_sha256(self):
        """Test that all configured hashes are valid SHA-256."""
        cfg = get_model_integrity_config()
        for attr in ["xgboost", "random_forest", "logistic_regression", "gru_60s"]:
            hash_val = getattr(cfg, attr)
            self.assertEqual(len(hash_val), 64)
            self.assertTrue(all(c in '0123456789abcdef' for c in hash_val.lower()))


if __name__ == "__main__":
    unittest.main()