# Reproducibility Guide — AI Network Intrusion Detection System V2

**Version:** 1.0  
**Date:** 2026-10-12  
**Repository State:** Commit `a51ae7c` (feat: integrate CICIDS temporal forecasting into Tab 7)

---

## 1. Environment & Dependencies

### Python Version
- **Required:** Python 3.11+ (tested on 3.11)
- **Project Python:** `.venv/Scripts/python` (Windows) or `python3` (Linux/macOS)

### Core Dependencies (from `requirements.txt` / `constraints.txt`)
| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | 2.5.2 | Numerical computing |
| `pandas` | 3.0.5 | Data manipulation |
| `scikit-learn` | 1.9.0 | ML preprocessing, metrics, baselines |
| `xgboost` | 3.4.1 | Gradient boosting (primary detector) |
| `torch` | 2.14.0+cpu | PyTorch (GRU forecasting) |
| `streamlit` | 1.63.0 | Dashboard framework |
| `plotly` | 7.0.0 | Interactive plots |
| `pyyaml` | 6.0.1 | Configuration |
| `pydantic` | 2.10.1 | Config validation |
| `pydantic-settings` | 2.7.1 | Env overrides |
| `joblib` | 1.6.0 | Model serialization |

### Exact Reproducible Install
```bash
# Clone
git clone https://github.com/Akashdeep-1/AI-Network-Intrusion-Detection-V2.git
cd AI-Network-Intrusion-Detection-V2

# Create environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install pinned dependencies
pip install -r requirements.txt
# OR for exact reproducibility:
pip install -c constraints.txt -r requirements.txt
```

---

## 2. Commands Quick Reference

### Dashboard
```bash
streamlit run app/app.py
# Opens http://localhost:8501
```

### Test Suite
```bash
# All tests
python -m unittest discover tests -v

# Specific suites
python -m unittest tests.test_pipeline_and_inference -v
python -m unittest tests.test_models -v
python -m unittest tests.test_upload_security -v
python -m unittest tests.test_model_integrity -v
python -m unittest tests.test_cicids_forecasting.TestCICIDSForecasting.test_timestamp_parsing -v
```

### Detection Pipeline (CICIoT2023)
```bash
# 1. Preprocess raw CSVs (chunked, deduplicate, map labels)
python -m src.data_preprocessing

# 2. Stratified split (80/20, zero overlap)
python -m src.split_data

# 3. Feature selection (36 features, remove IPv/LLC/Tot_size)
python -m src.feature_selection

# 4. Train models
python -m src.train_xgboost
python -m src.train_random_forest
python -m src.train_model  # Logistic Regression

# 5. Evaluate
python -m src.evaluate_model --all
```

### Forecasting Pipeline (CICIDS-2017)
```bash
# Complete: data prep → temporal split → GRU training → baseline evaluation
python -m src.forecasting.cicids_train
```

---

## 3. Dataset Preparation

### CICIoT2023 (Detection)
| Step | Action | Output |
|------|--------|--------|
| **Download** | From CIC: https://www.unb.ca/cic/datasets/iot-dataset-2023.html | `CICIoT2023.zip` |
| **Extract** | To `data/raw/ciciot2023/MERGED_CSV/` | 63 `Merged*.csv` files |
| **Preprocess** | `python -m src.data_preprocessing` | `data/processed/ciciot2023/ciciot2023_clean.csv` + `ciciot2023_balanced.csv` |
| **Split** | `python -m src.split_data` | `train_selected.csv`, `test_selected.csv` |
| **Feature Select** | `python -m src.feature_selection` | `selected_features.txt` (36 features) |

**Expected artifacts after prep:**
```
data/processed/ciciot2023/
├── ciciot2023_clean.csv          # 577,217 deduped rows
├── ciciot2023_balanced.csv       # Class-sampled
├── train_selected.csv            # 80% (461,773 rows)
├── test_selected.csv             # 20% (115,444 rows)
├── selected_features.txt         # 36 feature names
├── numeric_imputation_stats.json # Train-only medians
├── split_metadata.json           # Split verification
└── forecast_temporal_meta.json   # Temporal metadata (if generated)
```

### CICIDS-2017 (Forecasting)
| Step | Action | Output |
|------|--------|--------|
| **Download** | From CIC: https://www.unb.ca/cic/datasets/ids-2017.html | Daily CSVs or `MachineLearningCSV.zip` |
| **Place** | `data/raw/cicids2017/` | `monday.csv`...`friday.csv` (or `MachineLearningCVE/*.csv`) |
| **Provenance** | Documented in | `data/raw/cicids2017/DATASET_PROVENANCE.json` |

**Note:** The pipeline uses `bvk` HuggingFace mirror (79 cols, truncated Timestamp `MM:SS.m`) and `bencorn` ML mirror (79 cols, no Timestamp). True wall-clock only in `GeneratedLabelledFlows.zip` (88 cols) — not extracted here.

**Expected artifacts after training:**
```
data/processed/cicids2017/
├── ForecastFeatureSchema.json    # 21-dim state schema
├── states_60s.csv                # 2,405 non-empty windows
├── states_30s.csv                # (if generated)
└── scaler.joblib                 # (if saved separately)

models/
├── cicids_gru_W10_K5_60s.pt      # GRU checkpoint
├── cicids_gru_W10_K5_30s.pt      # (if generated)
└── cicids_lstm_W10_K5_60s.pt     # (if generated)

reports/
├── cicids_forecast_metrics_W10_K5_60s.json  # GRU vs persistence
├── cicids_forecast_decay_W10_K5_60s.csv     # Horizon decay
└── forecast_metrics.json          # Legacy (CICIoT2023 sequence)
```

---

## 4. Evaluation Metrics Generation

### Detection (CICIoT2023)
```bash
python -m src.evaluate_model --all
```
**Outputs:**
- `reports/xgboost_eval_report.txt` — XGBoost metrics + confusion matrix
- `reports/random_forest_eval_report.txt` — RF metrics
- `reports/logistic_regression_eval_report.txt` — LR metrics
- `reports/xgboost_feature_importance.csv` — Top features
- `reports/xgboost_confusion_matrix.csv` — Per-class confusion matrix
- `reports/xgboost_metrics.json` — JSON summary (if implemented)

### Forecasting (CICIDS-2017)
```bash
python -m src.forecasting.cicids_train
```
**Outputs:**
- `reports/cicids_forecast_metrics_W10_K5_60s.json` — Full GRU vs persistence (state MSE, binary ROC-AUC)
- `reports/cicids_forecast_decay_W10_K5_60s.csv` — Horizon decay table
- `reports/cicids_forecast_metrics_W10_K5_30s.json` (if 30s run)
- `reports/cicids_forecast_metrics_W10_K5_120s.json` (if 120s run)

---

## 5. Model Artifacts & Integrity Verification

### Artifact Locations & SHA-256 Hashes
| Artifact | Path | SHA-256 (Trusted) |
|----------|------|-------------------|
| XGBoost | `models/xgboost_model.joblib` | `539a5d661eb36863673efdebe0666d74416031850160c56eddb442a3cc4e0984` |
| Random Forest | `models/random_forest_model.joblib` | `08ed6ed2104e3c54af181b49fcd622dc66ea967811e4bac74fd8425786f1da2c` |
| Logistic Regression | `models/logistic_regression_model.joblib` | `8fe753b158853f8381db14f70995a4dc005bcb5f66977e8030c844597c30c9f1` |
| GRU (60s) | `models/cicids_gru_W10_K5_60s.pt` | `f56c9e09a42002fdce71a237fbe8b4d053ef46b45337b98c2cad0b39dba40f34` |

**Verification is automatic on app startup:**
```python
from src.model_integrity import load_verified_joblib, load_verified_torch, verify_all_artifacts

# CLI verification
python -c "from src.model_integrity import verify_all_artifacts; print(verify_all_artifacts())"
# Output: {'xgboost': True, 'random_forest': True, 'logistic_regression': True, 'gru_60s': True}
```

### Hash Storage
- **Config:** `config.yaml:model_integrity` section
- **Code:** `src/config.py:ModelIntegrityConfig` + `get_model_integrity_config()`
- **Verification:** `src/model_integrity.py` — hash checked **before** `joblib.load()` / `torch.load()`

---

## 6. Controlled Model Replacement Procedure

> **Only execute when intentionally replacing a trained model artifact.**

### Step 1: Train New Model
```bash
# Example: Retrain XGBoost with new hyperparameters
python -m src.train_xgboost
# Produces new models/xgboost_model.joblib
```

### Step 2: Compute New Hash
```bash
python -c "
import hashlib
h = hashlib.sha256()
with open('models/xgboost_model.joblib', 'rb') as f:
    for chunk in iter(lambda: f.read(8192), b''):
        h.update(chunk)
print(h.hexdigest())
"
```

### Step 3: Update Config
Edit `config.yaml`:
```yaml
model_integrity:
  xgboost: "NEW_SHA256_HERE"
  # ... other hashes unchanged
```

### Step 4: Verify
```bash
python -c "from src.model_integrity import verify_all_artifacts; print(verify_all_artifacts())"
# Should return all True
```

### Step 5: Run Full Test Suite
```bash
python -m unittest discover tests -v
# All 39 tests must pass
```

### Step 6: Commit
```bash
git add config.yaml models/xgboost_model.joblib
git commit -m "model: update XGBoost artifact (hash: NEW_SHA...)"
```

> ⚠️ **Never** update hashes without retraining. Hashes are the **source of truth** for artifact integrity.

---

## 7. Known Reproducibility Gaps

| Gap | Status | Impact |
|-----|--------|--------|
| **Raw dataset download date/checksum** | ❌ Not recorded | Cannot verify exact CICIoT2023/CICIDS-2017 version used |
| **CICIDS-2017 GeneratedLabelledFlows** | ❌ Not extracted | True wall-clock timestamps unavailable; B_DERIVED used |
| **Random seeds for PyTorch** | ⚠️ Partial | `torch.manual_seed(42)` not explicitly set in all paths |
| **CUDA non-determinism** | N/A | CPU-only (`torch.cpu`); no GPU used |
| **Random Forest / LR seeds** | ✅ Set | `random_state=42` in training scripts |
| **XGBoost seed** | ✅ Set | `random_state=42` |
| **Data shuffle seeds** | ✅ Set | `random_state=42` in `train_test_split` |
| **GRU training seed** | ❓ Unknown | Not explicitly documented in `cicids_train.py` |
| **Environment snapshot** | ❌ Missing | No `conda env export` / `pip freeze --all` in repo |
| **Cross-platform paths** | ✅ Pathlib | Windows/macOS/Linux compatible |

### Unverified Information (Marked Unknown)
- Exact CICIoT2023 release version used
- Exact CICIDS-2017 release version used
- Whether `GeneratedLabelledFlows.zip` was ever available locally
- Exact wall-clock training duration for GRU (reported: 9.6s in metrics)
- Whether any manual hyperparameter tuning occurred beyond reported values

---

## 8. Security & Scientific Limitations

### Security
| Limitation | Mitigation |
|------------|------------|
| `joblib.load()` executes code | SHA-256 verified **before** load; fails fast |
| `torch.load(weights_only=False)` required | Hash verified first; only trusted artifacts loaded |
| No model signing / supply chain attestation | Hashes in `config.yaml` (trusted input) |
| No adversarial robustness testing | Not claimed; research artifact only |
| Uncalibrated confidence scores | Documented; no auto-actions |

### Scientific
| Limitation | Honest Assessment |
|------------|-------------------|
| **CICIoT2023 has no timestamps** | Row order ≠ time; no temporal forecasting possible |
| **CICIDS-2017 ML CSVs lack Timestamp** | B_DERIVED row-order proxy; not true wall-clock |
| **Persistence beats GRU** | Δ MSE = -0.6675; reported honestly |
| **ROC-AUC 0.567** | Near-random; temporal label shift (Fri 99.9% attacks) |
| **No cross-dataset validation** | CICIoT2023 ≠ CICIDS-2017; no generalization claimed |
| **No adversarial evaluation** | Not a robustness benchmark |
| **Uncalibrated confidence** | `predict_proba` used as-is; no threshold actions |

---

## 9. Test Suite — Actual Results

### Full Suite (39 tests)
```bash
python -m unittest discover tests -v
# 39 tests in ~10s → ALL PASS
```

### Test Breakdown
| Suite | Tests | Focus |
|-------|-------|-------|
| `test_pipeline_and_inference` | 4 | Schema validation, dedup, split disjointness |
| `test_models` | 6 | Model loading, schema rejection, inference |
| `test_upload_security` | 13 | Format detection, size/row/col limits, malformed CSV |
| `test_model_integrity` | 13 | SHA-256 verification, tamper detection, hash config |
| `test_cicids_forecasting` (subset) | 3 | Timestamp parsing, persistence baseline, schema |

**All 39 tests PASS** (verified 2026-10-12).

---

## 10. Dashboard Verification

### Launch
```bash
streamlit run app/app.py
# → http://localhost:8501
```

### Tabs Verified
| Tab | Function | Dataset |
|-----|----------|---------|
| 1. Overview | Threat status, distribution, confidence | CICIoT2023 (upload) |
| 2. Traffic Analysis | Flow table, pagination | CICIoT2023 (upload) |
| 3. Threat Detection | Malicious flows + MITRE tags | CICIoT2023 (upload) |
| 4. Attack Distribution | Per-class breakdown | CICIoT2023 (upload) |
| 5. Feature Analysis | XGBoost importance + flow inspection | CICIoT2023 (upload) |
| 6. Data Quality | Schema compliance, imputation stats | CICIoT2023 (upload) |
| 7. Forecasting | **CICIDS-2017 precomputed demo** | **NOT upload** |

### Tab 7 Scope Banner (Verified Present)
> 🔬 **Research Demonstration — CICIDS-2017 Historical Forecasting**  
> • CICIoT2023 upload (Tabs 1–6) **NOT consumed**  
> • Persistence beats GRU (MSE 0.046 vs 0.713)  
> • All results precomputed from CICIDS-2017 historical data

---

## 11. Summary: What Can Be Reproduced Today

| Pipeline | From Scratch | From Artifacts |
|----------|--------------|----------------|
| **Detection (CICIoT2023)** | ✅ Full (raw CSV → metrics) | ✅ Inference only (model artifacts) |
| **Forecasting (CICIDS-2017)** | ✅ Full (raw CSV → metrics) | ✅ Inference only (GRU + states) |
| **Dashboard** | ✅ `streamlit run app/app.py` | ✅ |
| **Tests** | ✅ All 39 pass | ✅ |
| **Integrity Verification** | ✅ Auto on startup | ✅ Manual `verify_all_artifacts()` |

---

## 11. Files Created/Modified This Milestone

| File | Type | Purpose |
|------|------|---------|
| `MODEL_CARD_DETECTION.md` | New | Detection model documentation |
| `MODEL_CARD_FORECASTING.md` | New | Forecasting model documentation |
| `REPRODUCIBILITY.md` | New | This guide |

**No model artifacts, datasets, evaluation reports, or source logic were modified.**

---

> **Verification Date:** 2026-10-12  
> **Test Result:** 39/39 tests pass  
> **Dashboard:** Running at `http://localhost:8501` (health: `ok`)  
> **Integrity:** All 4 artifacts verified (SHA-256 matches `config.yaml`)  
> **No model weights, datasets, or evaluation metrics were altered.**