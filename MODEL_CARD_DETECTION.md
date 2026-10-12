# Model Card — CICIoT2023 Intrusion Detection

**Version:** 1.0  
**Date:** 2026-10-12  
**Repository:** AI Network Intrusion Detection System V2  
**Status:** Research artifact — not production-ready

---

## 1. Model Overview

### Models Included
| Model | Algorithm | Version | Artifact | SHA-256 |
|-------|-----------|---------|----------|---------|
| **XGBoost** (Primary) | XGBoost 3.4.1 | 1.0 | `models/xgboost_model.joblib` | `539a5d661eb36863673efdebe0666d74416031850160c56eddb442a3cc4e0984` |
| **Random Forest** | scikit-learn 1.9.0 | 1.0 | `models/random_forest_model.joblib` | `08ed6ed2104e3c54af181b49fcd622dc66ea967811e4bac74fd8425786f1da2c` |
| **Logistic Regression** | scikit-learn 1.9.0 | 1.0 | `models/logistic_regression_model.joblib` | `8fe753b158853f8381db14f70995a4dc005bcb5f66977e8030c844597c30c9f1` |

All artifacts are integrity-verified via SHA-256 before deserialization (see `src/model_integrity.py`).

---

## 2. Intended Use

### Supported Use Cases
- **Offline batch classification** of pre-extracted CICIoT2023 network flow features (36 features)
- **Research benchmarking** of classical ML algorithms on IoT intrusion detection
- **Educational demonstration** of intrusion detection pipeline with leakage controls
- **Comparative analysis** of XGBoost vs Random Forest vs Logistic Regression

### Unsupported / Prohibited Uses
- ❌ **Real-time / live network monitoring** — no packet capture, no stream processing
- ❌ **Production SOC deployment** — no alerting, no ticketing, no model monitoring, no authentication
- ❌ **Cross-dataset generalization** — trained on CICIoT2023 only; does not apply to CICIDS-2017 or other datasets
- ❌ **Zero-day / novel attack detection** — only detects 9 known categories from training data
- ❌ **Security-critical decisions without human review** — confidence scores are uncalibrated

---

## 3. Training Dataset & Provenance

| Aspect | Detail |
|--------|--------|
| **Dataset** | CICIoT2023 (Canadian Institute for Cybersecurity) |
| **Source** | https://www.unb.ca/cic/datasets/iot-dataset-2023.html |
| **Raw Files** | 63 merged CSVs in `data/raw/ciciot2023/MERGED_CSV/` |
| **Total Raw Rows** | ~1.2M (before cleaning) |
| **Clean Rows** | 577,217 (after deduplication & NaN removal) |
| **Train / Test Split** | 80 / 20 stratified, **zero feature-vector overlap** (hash-verified) |
| **Class Balance** | Class-aware sampling to targets (Benign: 100K, DDoS: 100K, DoS: 100K, Mirai: 100K, Recon: 75K, Spoofing: 75K, WebAttack: 50K, BruteForce: 25K, OtherAttack: 25K) |
| **Random Seed** | 42 (all splits) |

> ⚠️ **Provenance gap:** Exact download date, CICIoT2023 release version, and checksum of raw ZIP not recorded in repository.

---

## 4. Input Feature Schema & Preprocessing

### 36 Features (Exact Order — `src/data_pipeline.py:EXPECTED_FEATURE_COLUMNS`)
```
Header_Length, Protocol Type, Time_To_Live, Rate,
fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number,
ack_flag_number, ece_flag_number, cwr_flag_number,
ack_count, syn_count, fin_count, rst_count,
HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC,
TCP, UDP, DHCP, ARP, ICMP, IGMP,
Tot sum, Min, Max, AVG, Std, IAT, Number, Variance
```

### Removed Features (3)
- `IPv` (constant ~0.99)
- `LLC` (constant ~0.99)
- `Tot size` (duplicate of `AVG`)

### Preprocessing Pipeline (`src/data_preprocessing.py` → `src/data_pipeline.py`)
1. **Chunked CSV read** (100K rows/chunk) → map 34 original labels → 9 high-level categories (`src/label_mapping.py`)
2. **Numeric coercion** → `inf`/`-inf` → `NaN`
3. **Drop rows** with any NaN in features or label
4. **Global deduplication** on 36-feature vector (prevents train/test leakage)
5. **Stratified 80/20 split** with hash verification (`train_hashes.isdisjoint(test_hashes)`)
6. **Median imputation stats** computed on train only (`data/processed/ciciot2023/numeric_imputation_stats.json`)
7. **Inference-time** → same median imputation + feature alignment + schema validation (`src/inference.py`)

---

## 5. Training & Evaluation Methodology

### Training Commands
```bash
python -m src.train_xgboost      # XGBoost
python -m src.train_random_forest # Random Forest
python -m src.train_model         # Logistic Regression (with StandardScaler)
```

### Evaluation Commands
```bash
python -m src.evaluate_model --all
```

### Hyperparameters

| Model | Key Hyperparameters |
|-------|---------------------|
| **XGBoost** | `n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective=multi:softprob, tree_method=hist, random_state=42` |
| **Random Forest** | `n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42` |
| **Logistic Regression** | `C=1.0, max_iter=1000, solver=lbfgs, random_state=42` (with `StandardScaler` fit on train) |

### Leakage Controls (Verified in Tests)
- ✅ **Zero feature-vector overlap** train/test (`test_split_is_deterministic_and_has_no_feature_overlap`)
- ✅ **Scaler fit only on train** (Logistic Regression)
- ✅ **Median imputation stats from train only**
- ✅ **Stratification preserves class distribution**
- ✅ **Label mapping applied before split** (no target leakage)

---

## 6. Evaluation Results (Held-Out Test: 113,767 flows)

### Overall Metrics

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|-------|----------|-----------------|--------------|----------|-------------|
| **XGBoost** | **79.60%** | **75.82%** | **68.22%** | **70.48%** | **79.01%** |
| **Random Forest** | 77.48% | 70.03% | 72.51% | 69.35% | 78.29% |
| **Logistic Regression** | 66.60% | 59.30% | 59.80% | 57.40% | 67.60% |

*Source: `reports/xgboost_eval_report.txt`, `random_forest_eval_report.txt`, `logistic_regression_eval_report.txt`*

### Per-Class Performance (XGBoost — Primary)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Benign | 0.68 | 0.88 | 0.77 | 19,954 |
| BruteForce | 0.77 | 0.40 | 0.52 | 2,502 |
| **DDoS** | **0.90** | **0.70** | **0.79** | 16,814 |
| **DoS** | **0.75** | **0.92** | **0.83** | 16,486 |
| **Mirai** | **1.00** | **1.00** | **1.00** | 19,896 |
| OtherAttack | 0.51 | 0.38 | 0.43 | 4,989 |
| Recon | 0.68 | 0.66 | 0.67 | 14,824 |
| **Spoofing** | **0.92** | **0.86** | **0.89** | 14,188 |
| WebAttack | 0.61 | 0.36 | 0.45 | 4,114 |

### Confusion Matrix (XGBoost — Rows=True, Cols=Pred)
```
Classes: Benign, BruteForce, DDoS, DoS, Mirai, OtherAttack, Recon, Spoofing, WebAttack
[[17583    88     0     0     0   502  1256   333   192]
 [  810   995     0     0     0    62   470    59   106]
 [    0     0 11765  5029    17     0     3     0     0]
 [    0     0  1324 15149    12     0     1     0     0]
 [    0     0     8     9 19878     0     0     1     0]
 [ 1233    50     0     0     0  1883  1486   141   196]
 [ 3654    98     0     0     0   804  9714   286   268]
 [ 1412    31     0     0     0    50   402 12136   157]
 [ 1040    27     0     0     0   424   947   215  1461]]
```

> **Notable:** Mirai achieves perfect scores (clean separability). WebAttack and BruteForce have lowest recall. Benign often confused with Recon/OtherAttack.

---

## 7. Known Limitations

| Limitation | Impact |
|------------|--------|
| **Class imbalance** | Benign (19,954) vs WebAttack (4,114) vs BruteForce (2,502) — minority classes under-represented |
| **Uncalibrated confidence** | `predict_proba` outputs used directly; no Platt/isotonic calibration; confidence ≠ true probability |
| **No temporal information** | CICIoT2023 has no reliable per-flow timestamps; row order ≠ time |
| **Fixed 9 categories** | Cannot detect attacks outside the 34→9 mapped labels |
| **Dataset shift** | Performance on post-2023 IoT traffic unknown; protocol distribution changes not modeled |
| **Feature engineering fixed** | 36 hand-crafted features; no learned representations |
| **No adversarial robustness testing** | Vulnerable to evasion via feature manipulation |
| **Single dataset** | No cross-dataset validation (CICIDS-2017, UNSW-NB15, etc.) |

---

## 8. Inference Behavior & Integrity

### Inference Pipeline (`src/inference.py`)
1. **Schema validation** — all 36 features required; extra columns ignored; Label/Attack_Category dropped
2. **Median imputation** — using train-derived stats (`numeric_imputation_stats.json`)
3. **Scaling** — Logistic Regression only (StandardScaler from train)
4. **Prediction** → class label + confidence (max `predict_proba`)
5. **Output** → DataFrame with `Predicted_Category`, `Confidence`, `Is_Attack`, `Attack_Status`

### Integrity Verification (`src/model_integrity.py`)
- SHA-256 computed **before** `joblib.load()`
- Expected hashes in `config.yaml:model_integrity`
- On mismatch → `IntegrityError` → app stops (`st.stop()`)
- No fallback to benign prediction

### Dashboard Integration (Tabs 1–6)
- Upload → format detection → CICIoT2023 only → validated → predict → 6-tab results
- CICIDS-2017 uploads routed to Tab 7 with explicit message

---

## 8. Security Considerations

| Risk | Mitigation |
|------|------------|
| **Model artifact tampering** | SHA-256 verified before deserialization; hashes in `config.yaml` (trusted input) |
| **Arbitrary code execution via joblib** | Hash verified **before** `joblib.load()`; fails fast on mismatch |
| **Unbounded CSV upload** | 200 MB limit (Streamlit default + app-level check); row/col limits |
| **Uncalibrated confidence misinterpretation** | Documented; no threshold-based auto-actions |
| **Adversarial feature manipulation** | Not tested; no input sanitization beyond schema validation |

### Responsible Use Guidance
- Treat confidence as **relative ranking only**, not absolute probability
- Always pair with **human analyst review** before blocking traffic
- Monitor for **concept drift** — retrain when protocol distributions shift
- Do not expose model endpoint without **authentication + rate limiting**

---

## 9. Artifact Locations

| Artifact | Path |
|----------|------|
| XGBoost model | `models/xgboost_model.joblib` |
| Random Forest model | `models/random_forest_model.joblib` |
| Logistic Regression model | `models/logistic_regression_model.joblib` |
| XGBoost eval report | `reports/xgboost_eval_report.txt` |
| Random Forest eval report | `reports/random_forest_eval_report.txt` |
| Logistic Regression eval report | `reports/logistic_regression_eval_report.txt` |
| Feature importance | `reports/xgboost_feature_importance.csv` |
| Confusion matrix | `reports/xgboost_confusion_matrix.csv` |
| Median imputation stats | `data/processed/ciciot2023/numeric_imputation_stats.json` |
| Selected features list | `data/processed/ciciot2023/selected_features.txt` |

---

> **Document Status:** This model card reflects the repository state as of 2026-10-12. All metrics sourced from `reports/*.txt` files. No metrics were invented or altered.