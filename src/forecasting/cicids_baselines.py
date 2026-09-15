"""
Baselines — persistence / majority / EMA statistical
For S(t) state forecasting and y_bin attack.
Must answer: Does LSTM beat copying S(t)?
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def persistence_baseline_state(S_next_true: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    """S_pred(t+k)=S(t) for all k. X_seq [N,W,21], S_next [N,K,21]"""
    last = X_seq[:,-1,:]  # [N,21]
    K = S_next_true.shape[1]
    pred = np.repeat(last[:,None,:], K, axis=1)  # [N,K,21]
    return pred

def ema_baseline_state(X_seq: np.ndarray, K: int, alpha: float=0.5) -> np.ndarray:
    """EMA of last W -> predict same EMA forward"""
    # simple: EWMA of X_seq along W, then repeat
    # X_seq [N,W,21]
    ema = X_seq[:,0,:]
    for w in range(1, X_seq.shape[1]):
        ema = alpha*X_seq[:,w,:] + (1-alpha)*ema
    return np.repeat(ema[:,None,:], K, axis=1)

def majority_baseline_bin(y_train_bin: np.ndarray, N: int, K: int) -> np.ndarray:
    """Predict most frequent bin in train for all horizons."""
    flat = y_train_bin.reshape(-1)
    maj = int(np.bincount(flat, minlength=2).argmax())
    return np.full((N,K), maj, dtype=np.int64)

def persistence_baseline_bin(y_bin_history: np.ndarray, X_seq_unused, K=None) -> np.ndarray:
    """y_pred(t+k)=last observed attack state: derived from last window's attack ratio >0.1?
    If y_bin_history not available, fallback to y_bin of last? We approximate via X_seq not having label,
    so persistence on labels is not allowed without leakage — instead we evaluate persistence on y_bin target's last window val:
    Use y_bin true last? For honest baseline we use y_bin persistence: predict last known y (t) = value at t.
    Since S(t) does NOT contain label by design, true persistence baseline for classification should be:
    y_pred = 0/1 based on whether avg attack in last window? But that would leak. So for pure baselines we report majority only,
    and for state-aware persistence we use S(t) -> no label. For evaluation we can compute label persistence as oracle ceiling (cheating) to show upper bound.
    Here we implement oracle label persistence for comparison (documented as optimistic).
    """
    raise NotImplementedError

def evaluate_state_mse(pred: np.ndarray, true: np.ndarray) -> dict:
    """MSE/MAE/R2 per horizon K"""
    # pred true [N,K,21]
    mse = ((pred-true)**2).mean(axis=(0,2))  # [K]
    mae = np.abs(pred-true).mean(axis=(0,2))
    # R2 per dim avg
    # overall R2
    ss_res = ((true-pred)**2).sum()
    ss_tot = ((true-true.mean(axis=0, keepdims=True))**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot!=0 else 0
    return {"mse_per_k": mse.tolist(), "mae_per_k": mae.tolist(), "r2_overall": float(r2),
            "mse_mean": float(mse.mean()), "mae_mean": float(mae.mean())}

def evaluate_bin(pred: np.ndarray, true: np.ndarray, prob: np.ndarray| None=None) -> dict:
    """pred true [N,K] int 0/1"""
    K=true.shape[1]
    out={}
    for k in range(K):
        acc = accuracy_score(true[:,k], pred[:,k])
        f1 = f1_score(true[:,k], pred[:,k], zero_division=0)
        out[f"t+{k+1}"] = {"acc": float(acc), "f1": float(f1)}
    # flatten over K
    out["flat_acc"] = float(accuracy_score(true.reshape(-1), pred.reshape(-1)))
    out["flat_f1"] = float(f1_score(true.reshape(-1), pred.reshape(-1), zero_division=0))
    if prob is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(true.reshape(-1), prob.reshape(-1)))
            out["pr_auc"] = float(average_precision_score(true.reshape(-1), prob.reshape(-1)))
        except Exception:
            pass
    return out

if __name__ == "__main__":
    # smoke
    X = np.random.randn(100,10,21).astype(np.float32)
    Y = np.random.randn(100,5,21).astype(np.float32)
    pred = persistence_baseline_state(Y, X)
    print(evaluate_state_mse(pred, Y))
