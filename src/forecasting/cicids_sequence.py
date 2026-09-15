"""
Sequence builder + temporal split for CICIDS S(t)
S(t-W+1..t) -> S(t+1..t+K) (Form A) and -> y(t+1..t+K) (Form B)
Chronological split: 70% train /15% val /15% test by time (per STEP 6/10)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL_SET = ["BENIGN","FTP-Patator","SSH-Patator","DoS slowloris","DoS Slowhttptest","DoS Hulk","DoS GoldenEye","Heartbleed","Web Attack - Brute Force","Web Attack - XSS","Web Attack - Sql Injection","Infiltration","Bot","PortScan","DDoS"]
# High-level 9 for compatibility where needed
HIGH_LEVEL = ["Benign","DDoS","DoS","BruteForce","WebAttack","Bot","Recon","Infiltration","Other"]
def map_label_high(l: str) -> str:
    s=str(l).strip()
    if s.upper()=="BENIGN": return "Benign"
    if s=="DDoS": return "DDoS"
    if s.startswith("DoS"): return "DoS"
    if s in ("FTP-Patator","SSH-Patator","Web Attack - Brute Force"): return "BruteForce"
    if s in ("Web Attack - XSS","Web Attack - Sql Injection"): return "WebAttack"
    if s=="Bot": return "Bot"
    if s=="PortScan": return "Recon"
    if s=="Infiltration": return "Infiltration"
    if s=="Heartbleed": return "Other"
    return "Other"
HIGH_TO_IDX = {c:i for i,c in enumerate(HIGH_LEVEL)}

def build_sequences(
    S: np.ndarray,  # [T, 21]
    attack_ratio: np.ndarray,  # [T]
    dominant_labels: np.ndarray,  # [T] str
    window_times: np.ndarray,  # [T] epoch start
    W: int = 10, K: int = 5, stride: int = 1,
    label_mode: str = "binary",  # binary | high | raw
) -> Dict[str, np.ndarray]:
    """
    Build sequences with NO leakage:
    X = S[t-W+1 .. t]  (inclusive t)
    y_state = S[t+1 .. t+K]
    y_attack_ratio = attack_ratio[t+1..t+K]
    y_attack_label = dominant_labels[t+1..t+K] mapped
    t indexed such that window_times align.
    """
    T = len(S)
    N = (T - W - K) // stride + 1
    if N <= 0:
        raise ValueError(f"Not enough states T={T} for W={W} K={K} (need {W+K})")
    X_seq = np.zeros((N, W, S.shape[1]), dtype=np.float32)
    S_next = np.zeros((N, K, S.shape[1]), dtype=np.float32)
    y_ratio = np.zeros((N, K), dtype=np.float32)
    y_bin = np.zeros((N, K), dtype=np.int64)  # 0 benign (<0.1), 1 attack
    y_high = np.zeros((N, K), dtype=np.int64)
    t_start = np.zeros(N, dtype=np.float64)
    t_target = np.zeros((N, K), dtype=np.float64)
    for n in range(N):
        i = n*stride
        X_seq[n] = S[i:i+W]
        S_next[n] = S[i+W:i+W+K]
        y_ratio[n] = attack_ratio[i+W:i+W+K]
        # binary threshold 0.1 attack ratio
        y_bin[n] = (attack_ratio[i+W:i+W+K] > 0.1).astype(np.int64)
        # high mapping
        labs = dominant_labels[i+W:i+W+K]
        y_high[n] = np.array([HIGH_TO_IDX[map_label_high(l)] for l in labs], dtype=np.int64)
        t_start[n] = window_times[i+W-1]  # last observed
        t_target[n] = window_times[i+W:i+W+K]
    return {
        "X_seq": X_seq, "S_next": S_next,
        "y_ratio": y_ratio, "y_bin": y_bin, "y_high": y_high,
        "t_start": t_start, "t_target": t_target,
        "W": np.array(W), "K": np.array(K)
    }

def temporal_split(
    seq: Dict[str, np.ndarray],
    train_ratio: float = 0.70, val_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    """Chronological split on N sequences (already time-ordered).
    Asserts max(train_time) < min(val_time) and split.
    """
    N = len(seq["X_seq"])
    n_train = int(N*train_ratio)
    n_val = int(N*val_ratio)
    n_test = N - n_train - n_val
    if n_test <= 0 or n_val<=0:
        raise ValueError(f"Not enough sequences {N} for split {train_ratio}/{val_ratio}")
    # time checks
    t = seq["t_start"]
    # ensure time-ordered
    if not np.all(np.diff(t) >= -1e-6):
        raise ValueError("Sequences not time-ordered")
    splits = {}
    for name, s, e in [("train",0,n_train),("val",n_train,n_train+n_val),("test",n_train+n_val,N)]:
        splits[f"X_{name}"] = seq["X_seq"][s:e]
        splits[f"S_next_{name}"] = seq["S_next"][s:e]
        splits[f"y_ratio_{name}"] = seq["y_ratio"][s:e]
        splits[f"y_bin_{name}"] = seq["y_bin"][s:e]
        splits[f"y_high_{name}"] = seq["y_high"][s:e]
        splits[f"t_start_{name}"] = seq["t_start"][s:e]
        splits[f"t_target_{name}"] = seq["t_target"][s:e]
    # leakage assertions (will be tested)
    assert splits["t_start_train"].max() < splits["t_start_val"].min(), "train/val time overlap"
    assert splits["t_start_val"].max() < splits["t_start_test"].min(), "val/test time overlap"
    # no index overlap implicit by slicing
    print(f"Split N={N} -> train {n_train} val {n_val} test {n_test} | time {t[0]:.0f} -> {t[-1]:.0f}")
    print(f"  train time {splits['t_start_train'][0]:.0f}..{splits['t_start_train'][-1]:.0f}")
    print(f"  val   time {splits['t_start_val'][0]:.0f}..{splits['t_start_val'][-1]:.0f}")
    print(f"  test  time {splits['t_start_test'][0]:.0f}..{splits['t_start_test'][-1]:.0f}")
    # class dist
    for name in ["train","val","test"]:
        print(f"  {name} y_bin mean {(splits[f'y_bin_{name}']==1).mean():.3f}, y_high dist {np.bincount(splits[f'y_high_{name}'].reshape(-1), minlength=len(HIGH_LEVEL)).tolist()}")
    splits["high_classes"] = np.array(HIGH_LEVEL)
    splits["W"] = seq["W"]; splits["K"] = seq["K"]
    return splits

if __name__ == "__main__":
    # quick demo with synthetic
    S = np.random.randn(2400,21).astype(np.float32)
    ar = np.random.rand(2400).astype(np.float32)
    labs = np.array(["BENIGN"]*2400)
    times = np.arange(2400, dtype=float)*60
    seq = build_sequences(S, ar, labs, times, W=10, K=5)
    splits = temporal_split(seq)
    print({k: v.shape for k,v in splits.items() if isinstance(v, np.ndarray)})
