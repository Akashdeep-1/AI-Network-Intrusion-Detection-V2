# SIH26153 Architecture — Network Attack Forecasting S(t)->S(t+k)

## Dual Pipeline (Preserved + Extended)

```
RAW CICIoT2023 (63 MERGED CSVs)
  ↓
clean_and_deduplicate() — 577,217 rows, 36 features — NO shuffle
  ├─→ Classic Path (preserved):
  │    split_data.py (stratified 80/20, overlap=0) → train_selected.csv
  │    train_model / RF / XGB → models/*.joblib → evaluate_model --all → 79.6% XGBoost
  │    inference.predict_dataframe() — X(t)→y(t) — Streamlit Tabs 1-6
  │
  └─→ Forecast Path (new, temporal):
       state_builder.py — window=10, horizon=5, stride=2
           S(t) = [W=10, 36] + y[t]  → 288,602 sequences  (forecast_sequences.npz)
       split_temporal.py — time-ordered 70/15/15 — max(train) < min(test) — NO shuffle
           → forecast_temporal_split.npz
       train_lstm.py — LSTM(36→96×2 layers→9 classes ×5 heads) CPU
           → models/lstm_forecaster.pt + scaler_forecaster.joblib
       evaluate_forecast.py — measured vs majority baseline, t+1..t+5 decay
           → reports/forecast_metrics.json
       app/forecast_view.py — Tab 7: Observed (blue) vs Predicted (red) + uncertainty band
```

## Temporal Leakage Prevention — Proof

- Classic: hash dedup before random split → overlap 0 (existing, preserved)
- Forecast: contiguous index slices `[0:202021) [202021:245311) [245311:288602)` — file-order proxy time, no shuffle, verified in split_temporal.py: `assert train_slice.stop <= val_slice.start`

## Genuine Forecasting — Not Shifting

- Classifier shift `y(t) shifted k` would give identical t+1..t+5 metrics (rejected).
- LSTM learns `f(S[t-W:t] → y[t+1:t+K])` autoregressive heads, evaluated on strictly future holdout.

## Honest Components

- Explainability: SHAP on XGBoost only (measured), LSTM gradients (future)
- MITRE: RULE_MAP (rule-based) vs model_confidence (predicted) — separated per attck_mapping.py
- Uncertainty: softmax entropy + MC Dropout — honesty via confidence decay with k.
