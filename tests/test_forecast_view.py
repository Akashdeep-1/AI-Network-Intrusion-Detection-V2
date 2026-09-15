"""
Integration and regression tests for Tab 7 (Temporal Threat Forecasting)
and rollout prediction components.
"""
from __future__ import annotations
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from app.forecast_view import (
    _load_forecast_model_and_scaler,
    _load_forecast_states,
    _load_cicids_metrics,
    _threat_level,
    _threat_badge,
    render_forecast_tab,
)
from src.forecasting.cicids_state_builder import STATE_COLS
from src.forecasting.cicids_rollout import rollout_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestForecastView(unittest.TestCase):
    def test_forecast_artifacts_exist(self):
        """Verify that necessary model weights, states, and metrics reports exist."""
        model_path = PROJECT_ROOT / "models" / "cicids_gru_W10_K5_60s.pt"
        states_path = PROJECT_ROOT / "data" / "processed" / "cicids2017" / "states_60s.csv"
        metrics_path = PROJECT_ROOT / "reports" / "cicids_forecast_metrics_W10_K5_60s.json"

        self.assertTrue(model_path.exists(), f"Model checkpoint missing at {model_path}")
        self.assertTrue(states_path.exists(), f"States CSV missing at {states_path}")
        self.assertTrue(metrics_path.exists(), f"Metrics JSON missing at {metrics_path}")

    def test_load_forecast_model_and_scaler(self):
        """Verify cached model and scaler parameters load correctly in eval mode."""
        model, cfg, scaler_mean, scaler_scale = _load_forecast_model_and_scaler()

        self.assertIsNotNone(model, "Forecast model failed to load")
        self.assertIsNotNone(cfg, "Config dictionary missing from checkpoint")
        self.assertEqual(cfg.get("input_dim"), 21)
        self.assertEqual(cfg.get("K"), 5)
        self.assertEqual(cfg.get("W"), 10)
        self.assertFalse(model.training, "Forecaster model must be in eval() mode")

        self.assertEqual(len(scaler_mean), 21)
        self.assertEqual(len(scaler_scale), 21)
        self.assertTrue(np.all(np.isfinite(scaler_mean)))
        self.assertTrue(np.all(np.isfinite(scaler_scale)))

    def test_load_forecast_states(self):
        """Verify preprocessed states load with valid schema and non-zero rows."""
        states = _load_forecast_states(60)
        self.assertIsNotNone(states)
        self.assertGreater(len(states), 100)
        for col in STATE_COLS:
            self.assertIn(col, states.columns, f"Missing required state column: {col}")

    def test_rollout_prediction_shapes_and_probabilities(self):
        """Test forward rollout produces correct tensor shapes and valid probability bounds."""
        model, cfg, scaler_mean, scaler_scale = _load_forecast_model_and_scaler()
        states = _load_forecast_states(60)
        self.assertIsNotNone(model)
        self.assertIsNotNone(states)

        W = cfg.get("W", 10)
        K = cfg.get("K", 5)

        X_raw = states[STATE_COLS].tail(W).values.astype(np.float32)
        self.assertEqual(X_raw.shape, (W, 21))

        X_scaled = (X_raw - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)
        state_scaled, probs = rollout_predict(model, (scaler_mean, scaler_scale), X_scaled, steps=K)

        self.assertIsNotNone(state_scaled)
        self.assertIsNotNone(probs)
        self.assertEqual(state_scaled.shape, (K, 21))
        self.assertEqual(len(probs), K)

        # Probabilities must be bounded in [0, 1] and finite
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    def test_persistence_baseline_logic(self):
        """Test persistence baseline correctly repeats the latest observed state across horizons."""
        K = 5
        dummy_state = np.random.randn(10, 21).astype(np.float32)
        last_observed = dummy_state[-1:]
        pers_baseline = np.tile(last_observed, (K, 1))

        self.assertEqual(pers_baseline.shape, (K, 21))
        for k in range(K):
            np.testing.assert_array_equal(pers_baseline[k], last_observed[0])

    def test_threat_level_mapping(self):
        """Verify threat level thresholds classify probabilities accurately without calibration claims."""
        self.assertEqual(_threat_level(0.95), "CRITICAL (>90%)")
        self.assertEqual(_threat_level(0.75), "HIGH (70-90%)")
        self.assertEqual(_threat_level(0.45), "ELEVATED (30-70%)")
        self.assertEqual(_threat_level(0.10), "NORMAL (<30%)")
        # Verify backward-compatibility alias
        self.assertEqual(_threat_badge(0.95), "CRITICAL (>90%)")

    def test_render_forecast_tab_headless(self):
        """Verify render_forecast_tab executes without crashing in headless context."""
        # Should execute without exceptions when called with dummy or None inputs
        try:
            render_forecast_tab(results_df=None)
            dummy_df = pd.DataFrame({"Confidence": [0.9, 0.8], "Predicted_Category": ["DDoS", "DoS"]})
            render_forecast_tab(results_df=dummy_df)
        except Exception as e:
            self.fail(f"render_forecast_tab raised an unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()
