# CICIDS-2017 Hybrid Forecasting — Method & Honest Benchmark

**Dataset A (Tabs 1–6): CICIoT2023 MERGED_CSV** — X(t)→y(t) current-state multiclass detection. 36 features, 79.6% XGBoost, leakage-controlled, tabs 1–6 frozen.

**Dataset B (Tab 7): CICIDS-2017 — S(t-W..t)→S(t+1..t+k) temporal forecasting.** Separate experimental purpose, not homogeneous dataset. Source documented below.

## 1 Source & Selection

- Official: https://www.unb.ca/cic/datasets/ids-2017.html (CIC, UNB). Paper: Sharafaldin et al. ICISSP 2018. 5-day capture 2017-07-03..07, 25-user behavior, attack schedule Mon Benign → Tue FTP/SSH Patator → Wed DoS (Slowloris/Hulk/GoldenEye/Heartbleed) → Thu Web BruteForce/XSS/SQLi + Infiltration → Fri Bot/PortScan/DDoS.
- Files selected: **MachineLearningCSV.zip (bencorn/HF mirror of 8 ML CSVs, 235MB, verified header = 79 cols, NO Timestamp)** + **bvk/HF per-day CSVs (5 files, 774MB, 88 cols, Timestamp truncated to MM:SS.m)**.
- **Verification before download (STEP 1):** Checked header for Timestamp — ML CSV lacks it (confirmed 79 cols Destination Port..Label), GeneratedLabelledFlows (88 cols) has true `Timestamp M/D/YYYY H:MM:SS` but 52GB repo truncated mirror not yet extracted; bvk's Timestamp = `56:34.2` truncated (classification B). License: free research (cite paper). Size raw 774MB bvk + 235MB ML CSV. Offline after download.

## 2 Timestamp Analysis — B Derived (Not A)

- **Full format attempted:** `%m/%d/%Y %I:%M:%S %p`, `%Y-%m-%d %H:%M:%S` etc via `cicids_loader._parse_full_timestamp`.
- **bvk result:** 100% `B_DERIVED_row_order_uniform_8h` — truncated MM:SS.m only, 47% out-of-order within 5000-row sample (increases 2636 vs decreases 2363). **Not TRUE wall-clock (A).** Unwrapping as hour offsets inflated span to 79k hours — rejected.
- **Chosen B derived strategy (documented):** per-day `row_order_uniform_8h` — `epoch = day_09:00 + (row_idx / (N-1))*28800 + (MM%60)*0.001`. Preserves capture order, gives correct window density (480 windows/day ×5 =2400 for 60s), jitter ≤0.06s cannot invert order. Overnight gaps masked per-day (no interpolation across days). **Classification B (defensible derived ordering), not A.** Full A TRUE requires GeneratedLabelledFlows download — noted as limitation.

## 3 Feature Harmonization — ForecastFeatureSchema

21-dim S(t) (see `ForecastFeatureSchema.json`): flow_count, packet_count, byte_volume, packets/sec, bytes/sec, avg_packet_size, flow_duration_mean, flow_iat_mean/std, fwd/bwd packets mean, syn/rst/ack/fin/psh_rate, unique_src/dst ips, unique_dst_ports, tcp/udp_ratio.

- **A direct (8):** Protocol 6/17, TCP/UDP flags, FIN/SYN/RST/PSH/ACK counts.
- **B transformable (10):** Rate≈Flow Packets/s, AVG≈Average Packet Size, IAT≈Flow IAT Mean, Tot sum≈Total Length Fwd Packet.
- **C dataset-specific (12+):** Dst Port, Src IP dec, Flow Duration, Init_Win_bytes_* (CICIDS-only).
- **D unavailable in B:** Header_Length, Time_To_Live (IoT-specific). **Not forced into same schema; ForecastFeatureSchema separate.**

## 4 Temporal State S(t)

`S(t) = 21-dim window aggregate [flow_count...udp_ratio] ∈ R^21` where `t` indexes consecutive windows per day from 09:00. Evaluated 30/60/120s; **primary 60s → 2405 states, ~873 flows/window, attack_ratio mean 0.21** (see `states_60s.csv`). Labels `attack_ratio`, `dominant_label` computed per window but **excluded from input** (checked via `STATE_COLS`).

## 5 Target Design & Two Formulations

**Not training historical→future row label.** Instead:

- **Formulation A (primary, state forecast):** `S(t-9..t) → S(t+1..5)` regression MSE 21 dims. Then derive attack_prob via separately? Here evaluated joint.
- **Formulation B (attack prob):** `S(t-9..t) → P(attack|t+k)` binary (threshold attack_ratio>0.1) + high-level 9-class. Compare on same split. Attack labels used for target/eval, never in input (tested `attack_ratio not in STATE_COLS`).

## 6 Temporal Split & Leakage Controls

**Chronological 70/15/15 on windows:** train 1673 (Mon–Wed), val 358 (Thu morning), test 360 (Thu aft–Fri) for 60s W10K5 → `max(train)<min(val)<min(test)` asserted. Sequence overlap check via slicing (no shared indices). **Window leakage:** X=S[t-W+1..t], y=S[t+1..t+K] disjoint. **Scaler leakage:** StandardScaler fit on train only (mean saved), applied to val/test (tested). **Target leakage:** `attack_ratio` not in `STATE_COLS` (test asserts).

## 7 Baselines (meaningful temporal)

1. **Majority** — most frequent y_bin train (0) → fails on Friday 99.9% attack test (0.06% acc). 2. **Persistence** — `Ŝ(t+k)=S(t)` (strong for this S(t), MSE 0.0456). 3. **EMA** — optional. 4. **Logistic** — classic X→y not compared to forecast. Must answer *Does temporal modeling beat persistence?* — reported honestly (see §8).

## 8 LSTM/GRU — Lightweight CPU

`GRU input 21 hidden 64 layers2 dropout0.2 horizon5 heads K MLPs` (no autoreg accumulation), Adam 1e-3, batch 64, early patience 7. Trained 12 epochs (9.6s) for 60s_both, 18 epochs (30.8s) for 30s_state on CPU. **Inference <5ms/seq.** Saved `cicids_gru_W10_K5_60s.pt` + scaler.

## 9 Multi-step & Measured Decay

One-shot K=5 heads; evaluated per horizon t+1..5 (log-scale MSE plot). **Honest measured values:**

- **60s both:** model MSE 0.713 mean (R² -8.77) vs persistence 0.0456 (R² 0.37) → **Δ MSE -0.667 (model worse)**, per-K persistence 0.026→0.076 vs model 0.67→0.68.
- **30s state:** model 0.792 vs persistence 0.0448 → Δ -0.747, same pattern.
- **Binary:** model acc 0.9994 F1 0.9997 but ROC 0.57 (random) vs majority 0.0006 — test 99.9% attacks inflates acc; ROC reveals no discrimination due to temporal label shift (train 18% → val 79% → test 99.9%). Reported as measured, not fabricated. Files `cicids_forecast_metrics_W10_K5_60s.json` + `cicids_forecast_decay_W10_K5_60s.csv`.

## 10 Early Warning

Method: threshold 0.5 on P(attack|t+k), first warning k_pred vs first true k_true → lead=(k_true−k_pred)*60s, evaluated only on benign→attack transition windows. **Not computing fabricated lead time** — CICIDS is scenario progression (isolated daily attacks), not continuous causal campaign. Dataset supports temporal evolution + scenario progression distinction; lead times reflect traffic evolution, not attacker causality. Implementation in `cicids_early_warning.py`.

## 11 Attack Progression vs Scenario

Day schedule is **scenario progression**, not single campaign. No claim of continuous campaign.

## 12 MITRE Mapping — Separate rule vs prediction

Direct/inferred/heuristic tags. Rule table vs model P(attack) shown separately. See dashboard Tab 7.

## 13 Required Changes

All new under `src/forecasting/cicids_*`, `data/processed/cicids2017/`, `tests/test_cicids_forecasting.py`, `docs/`; Tab 7 upgraded to real interface with observed (blue) vs forecast (red) vs persistence (gray). Classic pipeline frozen.

## 14 Limitations & Honest Risks

- B derived, not A TRUE — full GeneratedLabelledFlows timestamp still needed for A.
- Persistence beats GRU for S(t) on this B data — negative result reported.
- Label shift makes binary accuracy misleading — ROC 0.57 reported honestly.
- Small sample 2405 windows vs previous 288k — may underfit.
