"""
Tests for CICIDS temporal forecasting — leakage, parsing, sequences
STEP 19 — unittest compatible
"""
import numpy as np
import pandas as pd
import unittest
from pathlib import Path

class TestCICIDSForecasting(unittest.TestCase):
    def test_timestamp_parsing(self):
        from src.forecasting.cicids_loader import _parse_full_timestamp, _parse_truncated, DAY_TO_DATE
        self.assertIsNotNone(_parse_full_timestamp("7/3/2017 3:56:34 PM"))
        self.assertIsNotNone(_parse_full_timestamp("2017-07-03 15:56:34"))
        self.assertIsNone(_parse_full_timestamp("56:34.2"))
        self.assertIsNotNone(_parse_truncated("56:34.2"))
        self.assertAlmostEqual(_parse_truncated("56:34.2"), 56*60+34.2)
        self.assertEqual(DAY_TO_DATE["monday"],"2017-07-03")

    def test_chronological_sort_and_quality(self):
        from src.forecasting.cicids_loader import load_all_cicids
        df = load_all_cicids()
        self.assertTrue(df["_epoch"].is_monotonic_increasing)
        self.assertTrue(df["_ts_quality"].isin(["B_DERIVED_row_order_uniform_8h","A_TRUE","B_DERIVED_truncated_reconstructed","C_PROXY_row_order"]).any())
        self.assertEqual(df["_epoch"].isna().sum(),0)
        self.assertIn("monday", set(df["_day"].unique()))
        self.assertIn("friday", set(df["_day"].unique()))

    def test_window_state_counts(self):
        from src.forecasting.cicids_loader import load_all_cicids
        from src.forecasting.cicids_state_builder import build_states
        df = load_all_cicids()
        for w in [30,60,120]:
            s, meta = build_states(df, window_sec=w)
            self.assertEqual(len(s), meta["nonempty_windows"])
            self.assertEqual(meta["window_sec"], w)
            self.assertGreater(s["flow_count"].min(), 0)
            expected = 5*8*3600/w
            self.assertAlmostEqual(len(s), expected, delta=50)

    def test_no_label_leakage_in_S(self):
        from src.forecasting.cicids_loader import load_all_cicids
        from src.forecasting.cicids_state_builder import build_states, STATE_COLS, states_to_arrays
        df = load_all_cicids()
        s,_ = build_states(df, window_sec=60)
        self.assertNotIn("attack_ratio", STATE_COLS)
        self.assertNotIn("dominant_label", STATE_COLS)
        S, ar, labs, cols = states_to_arrays(s)
        self.assertEqual(S.shape[1],21)
        self.assertNotIn("attack_ratio", cols)

    def test_temporal_split_no_leakage(self):
        from src.forecasting.cicids_loader import load_all_cicids
        from src.forecasting.cicids_state_builder import build_states, states_to_arrays
        from src.forecasting.cicids_sequence import build_sequences, temporal_split
        df = load_all_cicids()
        s,_ = build_states(df, window_sec=60)
        S, ar, labs,_ = states_to_arrays(s)
        times = s["_window_start"].values.astype(float)
        seq = build_sequences(S, ar, labs, times, W=10, K=5)
        splits = temporal_split(seq, train_ratio=0.7, val_ratio=0.15)
        self.assertLess(splits["t_start_train"].max(), splits["t_start_val"].min())
        self.assertLess(splits["t_start_val"].max(), splits["t_start_test"].min())
        self.assertEqual(len(splits["X_train"])+len(splits["X_val"])+len(splits["X_test"]), len(seq["X_seq"]))
        self.assertTrue((splits["t_target_train"][:,0] > splits["t_start_train"]).all())

    def test_scaler_leakage_only_train(self):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from src.forecasting.cicids_loader import load_all_cicids
        from src.forecasting.cicids_state_builder import build_states, states_to_arrays
        from src.forecasting.cicids_sequence import build_sequences, temporal_split
        df = load_all_cicids()
        s,_ = build_states(df, window_sec=60)
        S, ar, labs,_ = states_to_arrays(s)
        times = s["_window_start"].values.astype(float)
        seq = build_sequences(S, ar, labs, times, W=10, K=5)
        splits = temporal_split(seq)
        scaler = StandardScaler()
        scaler.fit(splits["X_train"].reshape(-1, splits["X_train"].shape[2]))
        self.assertFalse(np.allclose(scaler.mean_, splits["X_val"].reshape(-1, splits["X_val"].shape[2]).mean(axis=0)))

    def test_sequence_shapes_and_horizon(self):
        from src.forecasting.cicids_loader import load_all_cicids
        from src.forecasting.cicids_state_builder import build_states, states_to_arrays
        from src.forecasting.cicids_sequence import build_sequences
        df = load_all_cicids()
        s,_ = build_states(df, window_sec=60)
        S, ar, labs,_ = states_to_arrays(s)
        times = s["_window_start"].values.astype(float)
        seq = build_sequences(S, ar, labs, times, W=10, K=5)
        self.assertEqual(seq["X_seq"].shape[1],10)
        self.assertEqual(seq["X_seq"].shape[2],21)
        self.assertEqual(seq["S_next"].shape[1],5)
        self.assertEqual(seq["y_bin"].shape[1],5)

    def test_persistence_baseline_and_model_loading(self):
        from src.forecasting.cicids_baselines import persistence_baseline_state, evaluate_state_mse
        X = np.random.randn(10,10,21).astype(np.float32)
        Y = np.random.randn(10,5,21).astype(np.float32)
        pred = persistence_baseline_state(Y, X)
        self.assertEqual(pred.shape, Y.shape)
        res = evaluate_state_mse(pred, Y)
        self.assertIn("mse_per_k", res)
        self.assertEqual(len(res["mse_per_k"]),5)
        pt = Path("models/cicids_gru_W10_K5_60s.pt")
        if pt.exists():
            import torch
            ckpt = torch.load(pt, map_location="cpu")
            self.assertIn("model_state", ckpt)
            self.assertIn("config", ckpt)

    def test_multistep_rollout_shape(self):
        from src.forecasting.cicids_rollout import rollout_predict, load_model
        pt = Path("models/cicids_gru_W10_K5_60s.pt")
        if not pt.exists():
            self.skipTest("model not trained yet")
        import torch, numpy as np
        from src.forecasting.cicids_loader import load_all_cicids
        from src.forecasting.cicids_state_builder import build_states, states_to_arrays
        from src.forecasting.cicids_sequence import build_sequences, temporal_split
        from sklearn.preprocessing import StandardScaler
        df = load_all_cicids()
        s,_ = build_states(df, window_sec=60)
        S, ar, labs,_ = states_to_arrays(s)
        times = s["_window_start"].values.astype(float)
        seq = build_sequences(S, ar, labs, times, W=10, K=5)
        splits = temporal_split(seq)
        scaler = StandardScaler(); scaler.fit(splits["X_train"].reshape(-1,21))
        X_test = scaler.transform(splits["X_test"][0:1].reshape(-1,21)).reshape(1,10,21).astype(np.float32)
        model, cfg, scaler_tuple = load_model(pt, device="cpu")
        state, prob = rollout_predict(model, scaler_tuple, X_test[0])
        self.assertTrue(state.shape==(5,21) or (prob is not None and prob.shape==(5,)))

    def test_forecast_output_schema(self):
        import json
        p = Path("reports/cicids_forecast_metrics_W10_K5_60s.json")
        if not p.exists():
            self.skipTest("metrics not yet generated")
        data = json.loads(p.read_text())
        self.assertIn("config", data)
        self.assertIn("results", data)
        self.assertTrue("state" in data["results"] or "bin" in data["results"])
        self.assertTrue(Path("reports/cicids_forecast_decay_W10_K5_60s.csv").exists())
