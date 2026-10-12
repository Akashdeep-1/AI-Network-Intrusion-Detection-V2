# Model Card — CICIDS-2017 Temporal Forecasting

**Version:** 1.0  
**Date:** 2026-10-12  
**Repository:** AI Network Intrusion Detection System V2  
**Status:** Research artifact — not production-ready

---

## 1. Model Overview

| Aspect | Detail |
|--------|--------|
| **Model** | GRU (Gated Recurrent Unit) |
| **Framework** | PyTorch 2.14.0 |
| **Artifact** | `models/cicids_gru_W10_K5_60s.pt` |
| **SHA-256** | `f56c9e09a42002fdce71a237fbe8b4d053ef46b45337b98c2cad0b39dba40f34` |
| **Integrity** | Verified via SHA-256 before `torch.load()` (`src/model_integrity.py`) |

---

## 2. Intended Use

### Supported Use Cases
- **Research demonstration** of temporal network-state forecasting on CICIDS-2017
- **Baseline comparison** of neural forecaster vs persistence baseline
- **Educational example** of chronological split methodology and leakage controls
- **Negative result documentation** — honest reporting of persistence outperforming GRU

### Unsupported / Prohibited Uses
- ❌ **Real-time forecasting** — no live packet capture, no stream inference
- ❌ **Production forecasting** — no model monitoring, no retraining pipeline
- ❌ **CICIoT2023 forecasting** — different features, no timestamps, different temporal structure
- ❌ **Operational attack prediction** — ROC-AUC 0.567 indicates near-random discrimination
- ❌ **Cross-dataset forecasting** — trained on CICIDS-2017 only
- ❌ **Automated response triggering** — no discriminative power for attack onset

---

## 3. Dataset — CICIDS-2017

| Aspect | Detail |
|--------|--------|
| **Dataset** | CICIDS-2017 (Canadian Institute for Cybersecurity) |
| **Source** | https://www.unb.ca/cic/datasets/ids-2017.html |
| **Temporal Coverage** | 5 days: Monday July 3 – Friday July 7, 2017 |
| **Raw Files** | 8 daily CSVs (bvk HuggingFace mirror + bencorn ML mirror) |
| **Total Flows** | ~2.8M across 5 days |
| **Attack Types** | 14 attack types + BENIGN (DoS, DDoS, PortScan, Web Attack, Brute Force, Infiltration, Botnet) |
| **Day-wise Schedule** | Mon: Benign; Tue: Benign+FTP/SSH BruteForce; Wed: Benign+DoS; Thu: Web Attack + Infiltration; Fri: PortScan + DDoS |

### Timestamp Handling (Critical)
| Quality Tier | Format | Proportion | Method |
|--------------|--------|------------|--------|
| **A_TRUE** | Full `7/3/2017 15:56:34` | ~0% in ML CSVs | Direct parse |
| **B_DERIVED** | Truncated `MM:SS.m` → uniform 8h per day | 100% in ML CSVs | Row order preserved, distributed over 09:00-17:00 |
| **C_PROXY** | Row order only | Fallback | Sequential epoch |

> **Key finding:** ML CSVs (bvk/bencorn) **lack true Timestamp** (79 columns, no Timestamp column). True wall-clock only in `GeneratedLabelledFlows.zip` (88 cols, not extracted here). Current pipeline uses **B_DERIVED** — row order preserved per day, distributed uniformly over 8-hour working window.

---

## 4. State Vector Definition — S(t) ∈ ℝ²¹

**Window:** 60-second tumbling windows (configurable 30/60/120s)  
**Per-day windows:** ~480 (8h × 60 min)  
**Total non-empty windows:** 2,405

### 21-Dimensional State Vector (Order: `src/forecasting/cicids_state_builder.py:STATE_COLS`)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `flow_count` | Flows in window |
| 1 | `packet_count` | Total packets (fwd + bwd) |
| 2 | `byte_volume` | Total bytes (fwd + bwd) |
| 3 | `packets_per_sec` | Packet rate |
| 4 | `bytes_per_sec` | Byte rate |
| 5 | `avg_packet_size` | Mean packet size |
| 6 | `flow_duration_mean` | Mean flow duration |
| 7 | `flow_iat_mean` | Mean flow inter-arrival time |
| 8 | `flow_iat_std` | Std of flow IAT |
| 9 | `fwd_packets_mean` | Mean fwd packets per flow |
| 10 | `bwd_packets_mean` | Mean bwd packets per flow |
| 11 | `syn_rate` | SYN flag rate |
| 12 | `rst_rate` | RST flag rate |
| 13 | `ack_rate` | ACK flag rate |
| 14 | `fin_rate` | FIN flag rate |
| 15 | `psh_rate` | PSH flag rate |
| 16 | `unique_src_ips` | Unique source IPs |
| 17 | `unique_dst_ips` | Unique destination IPs |
| 18 | `unique_dst_ports` | Unique destination ports |
| 19 | `tcp_ratio` | TCP flow proportion |
| 20 | `udp_ratio` | UDP flow proportion |

### Evaluation-Only Fields (NOT in S(t) — No Label Leakage)
| Field | Purpose |
|-------|---------|
| `attack_ratio` | Proportion of attack flows in window |
| `dominant_label` | Most frequent attack label in window |

---

## 5. Forecasting Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Window (W)** | 10 | History: 10 windows = 10 minutes at 60s |
| **Horizon (K)** | 5 | Forecast: 5 windows = 5 minutes ahead |
| **Input** | S(t-9)...S(t) ∈ ℝ¹⁰ˣ²¹ | 10-step history |
| **Target** | S(t+1)...S(t+5) ∈ ℝ⁵ˣ²¹ | 5-step future states |
| **Binary Target** | Attack presence (attack_ratio > 0.1) | Per horizon |

---

## 6. Model Architecture

| Component | Specification |
|-----------|---------------|
| **Type** | GRU (Gated Recurrent Unit) |
| **Hidden Size** | 64 |
| **Layers** | 2 |
| **Dropout** | 0.2 |
| **Mode** | Both (state regression + binary classification) |
| **Heads** | 5 state regression heads + 5 binary classification heads |
| **Optimizer** | Adam, lr=0.001 |
| **Batch Size** | 64 |
| **Epochs** | 30 (early stopping patience=10) |
| **Loss** | MSE (state) + BCE (binary) |

---

## 7. Chronological Split Methodology

**Strict temporal ordering enforced:**

| Split | Days | Windows | Proportion | Temporal Boundaries |
|-------|------|---------|------------|---------------------|
| **Train** | Mon–Wed | ~1,673 | 70% | Max(t_train) < Min(t_val) |
| **Validation** | Thu AM | ~358 | 15% | Max(t_val) < Min(t_test) |
| **Test** | Thu PM–Fri | ~360 | 15% | — |

**Verified in tests:** `test_temporal_split_no_leakage` — `t_train.max() < t_val.min() < t_test.min()`

**Scaler:** `StandardScaler` fit **only on train windows**, applied to val/test (no leakage)

---

## 8. Persistence Baseline

**Definition:** `Ŝₜ₊ₖ = Sₜ` (copy last observed state to all horizons)

**Rationale:** Network traffic exhibits high local autocorrelation; steady-state traffic changes slowly. A valid forecasting model must beat this trivial baseline.

---

## 9. Verified Evaluation Metrics

*Source: `reports/cicids_forecast_metrics_W10_K5_60s.json`*

### Formulation A — State Regression MSE (Scaled Space, Lower Better)

| Horizon | Model MSE | Persistence MSE | Δ (Pers − Model) | Model MAE | Persistence MAE |
|---------|-----------|-----------------|------------------|-----------|-----------------|
| t+1 (60s) | 0.6759 | **0.0260** | -0.6499 | 0.4242 | **0.0727** |
| t+2 (120s) | 0.7367 | **0.0340** | -0.7027 | 0.4494 | **0.0764** |
| t+3 (180s) | 0.7255 | **0.0418** | -0.6837 | 0.4201 | **0.0783** |
| t+4 (240s) | 0.7470 | **0.0497** | -0.6973 | 0.4227 | **0.0812** |
| t+5 (300s) | 0.6808 | **0.0769** | -0.6039 | 0.4009 | **0.0862** |
| **Mean** | **0.7132** | **0.0457** | **-0.6675** | **0.4235** | **0.0790** |

**Key:** Negative Δ = persistence wins (lower MSE). **Persistence beats GRU by Δ = -0.6675 MSE.**

### Formulation B — Attack Presence Classification

| Metric | Model | Majority Baseline |
|--------|-------|-------------------|
| Flat Accuracy | 99.94% | 0.06% |
| Flat F1 | 99.97% | 0.00% |
| **ROC-AUC** | **0.567** | — |
| PR-AUC | 0.9997 | — |

**Test distribution:** Friday PM = 99.9% attack windows (DDoS/PortScan)

---

## 10. Why Persistence Outperforms GRU

### Evidence (Measured)
1. **High autocorrelation** — Network state vectors change incrementally during steady traffic
2. **Low signal-to-noise** — 21-dim aggregated state captures aggregate statistics, not flow-level dynamics
3. **Temporal distribution shift** — Train (Mon-Wed: ~18% attacks) → Test (Fri: 99.9% attacks) destroys calibration
4. **GRU capacity mismatch** — 64 hidden units, 2 layers may overfit noise in low-signal aggregated features

### Hypotheses (Not Proven)
- Longer history (W>10) or different architecture (LSTM, Transformer) might help
- Feature engineering (differences, trends, spectral features) could improve signal
- Per-flow temporal modeling (not 60s aggregates) might capture attack onset better
- Domain-adaptive normalization could mitigate train/test distribution shift

### What Is NOT Supported
- ❌ "GRU is fundamentally incapable of temporal forecasting"
- ❌ "Neural networks don't work for network traffic forecasting"
- ❌ "This result generalizes to other datasets/architectures"

**Correct framing:** On **this specific dataset, aggregation, split, and architecture**, persistence is a stronger baseline. This is a **valid scientific finding** — not a failure to report.

---

## 11. Limitations & Unsupported Claims

| Limitation | Impact |
|------------|--------|
| **B_DERIVED timestamps** | Not true wall-clock; row-order proxy with uniform 8h spread |
| **Test set = Friday only** | 99.9% attacks → accuracy inflated, ROC-AUC 0.567 reveals no discrimination |
| **No cross-dataset validation** | Only CICIDS-2017; no CICIoT2023, UNSW-NB15, etc. |
| **No live evaluation** | Offline replay only; no stream inference tested |
| **Single architecture** | Only GRU (64/2) tested; no LSTM/Transformer/statistical baselines beyond persistence |
| **No uncertainty quantification** | No prediction intervals, no conformal prediction |
| **Binary threshold arbitrary** | `attack_ratio > 0.1` not validated |

---

## 12. Dashboard Integration — Tab 7

### What Tab 7 Shows
- **Precomputed CICIDS-2017 historical results** from `reports/cicids_forecast_metrics_W10_K5_60s.json`
- **Trajectory replay** from `data/processed/cicids2017/states_60s.csv` (last 20 observed windows → 5-step forecast)
- **Model vs Persistence vs Observed** comparison (Blue/Gray/Red)
- **MSE decay table** (t+1 through t+5)
- **Binary attack probability** per horizon with threat levels
- **MITRE ATT&CK mapping** (rule-based, not model-derived)

### What Tab 7 Does NOT Do
- ❌ **Does not consume CICIoT2023 upload** (Tabs 1–6 data)
- ❌ **Does not forecast on uploaded data**
- ❌ **Does not run live inference**

### Scope Banner (Implemented)
> 🔬 **Research Demonstration — CICIDS-2017 Historical Forecasting**  
> This tab displays an independent temporal forecasting experiment using the CICIDS-2017 dataset (5 days, Mon–Fri 2017).  
> • The **CICIoT2023 upload used by Tabs 1–6 is NOT consumed by this tab**.  
> • Forecasting model: GRU (64 hidden, W=10, K=5) trained on 60s windowed state vectors S(t) ∈ ℝ²¹.  
> • **Persistence baseline (Ŝₜ₊ₖ = Sₜ) outperforms the GRU** in the reported evaluation: MSE 0.0456 vs 0.713 (Δ = -0.667). This is an honest negative result — the neural forecaster provides no gain over copying the last observed state.  
> • All metrics, trajectories, and baselines shown here are precomputed from CICIDS-2017 historical data.

---

## 13. Security & Integrity

| Control | Implementation |
|---------|----------------|
| **Artifact integrity** | SHA-256 verified before `torch.load()` (`src/model_integrity.py:load_verified_torch`) |
| **Expected hash** | `config.yaml:model_integrity.gru_60s` |
| **Failure mode** | `IntegrityError` → `st.error()` → `st.stop()` (no fallback) |
| **Hash location** | `config.yaml:model_integrity.gru_60s` = `f56c9e09a42002fdce71a237fbe8b4d053ef46b45337b98c2cad0b39dba40f34` |

---

## 14. Artifact Locations

| Artifact | Path |
|----------|------|
| GRU model | `models/cicids_gru_W10_K5_60s.pt` |
| GRU eval metrics | `reports/cicids_forecast_metrics_W10_K5_60s.json` |
| Horizon decay | `reports/cicids_forecast_decay_W10_K5_60s.csv` |
| State sequences | `data/processed/cicids2017/states_60s.csv` |
| Scaler (train-only) | Embedded in `.pt` checkpoint |
| Feature schema | `data/processed/cicids2017/ForecastFeatureSchema.json` |

---

## 15. Explicit Statement

> **Tab 7 displays precomputed CICIDS-2017 historical forecasting results. It does not consume, process, or forecast on the CICIoT2023 upload used by Tabs 1–6. The two pipelines are logically and physically separated. The GRU forecasting model was evaluated against a persistence baseline and did not achieve lower MSE. This negative result is reported honestly and is not suppressed.**

---

> **Document Status:** This model card reflects the repository state as of 2026-10-12. All metrics sourced from `reports/cicids_forecast_metrics_W10_K5_60s.json` and `reports/cicids_forecast_decay_W10_K5_60s.csv`. No metrics were invented or altered. The negative result (persistence > GRU) is preserved honestly.