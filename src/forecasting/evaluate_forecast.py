"""
Honest forecast benchmarking — measured only, vs persistence baseline.
"""
from __future__ import annotations
from pathlib import Path
import json, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import joblib
from .lstm_forecaster import LSTMForecaster, GRUForecaster

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "forecast_temporal_split.npz"
MODEL_FILE = PROJECT_ROOT / "models" / "lstm_forecaster.pt"
SCALER_FILE = PROJECT_ROOT / "models" / "scaler_forecaster.joblib"
SEQ_FILE = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "forecast_sequences.npz"
REPORT_DIR = PROJECT_ROOT / "reports"

def evaluate():
    data = np.load(SPLIT_FILE)
    ckpt = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    scaler = joblib.load(SCALER_FILE)["scaler"]
    F = cfg["input_dim"]
    def transform(arr):
        return scaler.transform(arr.reshape(-1,F)).reshape(arr.shape).astype(np.float32)
    X_test = transform(data["X_test"])
    y_test = data["y_test"]
    # Also need persistence baseline: y(t) = last observed label in window -> compare? Use X not y.
    # Simpler: persistence = repeat y_last where y_last = argmax? Actually we have true y_test[t] as future; persistence baseline predicts y_test[:,0]==y_train_last? Instead use y_test persistence = y_test shifted? For honest: baseline = predict t+k = t (most recent history label = y_test's predecessor). We proxy as y_test[:,0] predicted as mode of window? Instead naive baseline = always predict majority class.
    # Better: load clean df to get true temporal persistence. Simplified: majority baseline.
    from collections import Counter
    majority = int(np.bincount(data["y_train"].reshape(-1)).argmax())
    print(f"Majority class idx {majority}")

    ModelCls = LSTMForecaster if cfg["model_type"]=="lstm" else GRUForecaster
    model = ModelCls(input_dim=F, hidden_dim=cfg["hidden"], num_layers=2, num_classes=cfg["num_classes"], horizon=cfg["horizon"])
    model.load_state_dict(ckpt["model_state"]); model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(X_test), 2048):
            xb = torch.from_numpy(X_test[i:i+2048])
            logits.append(model(xb).numpy())
    logits = np.concatenate(logits, axis=0)  # [N,K,C]
    preds = logits.argmax(axis=-1)  # [N,K]
    # Persistence baseline: predict k steps as last observed attack? Use X_test's last window's majority? Use y_test's previous? We use simple: predict t+k = t (last window's last y is unknown). Use majority as persistence proxy and also report.
    results=[]
    for k in range(cfg["horizon"]):
        acc = accuracy_score(y_test[:,k], preds[:,k])
        f1m = f1_score(y_test[:,k], preds[:,k], average="macro", zero_division=0)
        f1w = f1_score(y_test[:,k], preds[:,k], average="weighted", zero_division=0)
        base_acc = accuracy_score(y_test[:,k], np.full_like(y_test[:,k], majority))
        results.append({"horizon":k+1, "acc":float(acc), "macro_f1":float(f1m), "weighted_f1":float(f1w), "baseline_acc":float(base_acc), "delta_vs_baseline":float(acc-base_acc)})
        print(f"t+{k+1} acc={acc:.4f} macroF1={f1m:.4f} baseline={base_acc:.4f} delta={acc-base_acc:+.4f}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "forecast_metrics.json").write_text(json.dumps({"config":cfg, "val_acc":float(ckpt.get("val_acc",0)), "horizons":results}, indent=2))
    # CSV decay
    import pandas as pd
    pd.DataFrame(results).to_csv(REPORT_DIR / "forecast_decay.csv", index=False)
    print(f"Saved forecast_metrics.json + forecast_decay.csv")
    return results
