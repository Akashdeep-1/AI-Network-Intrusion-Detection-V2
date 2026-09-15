"""
Train LSTM forecaster on temporal split — CPU only.
"""
from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from .lstm_forecaster import LSTMForecaster, GRUForecaster

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "forecast_temporal_split.npz"
SEQ_FILE = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "forecast_sequences.npz"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

def train(model_type="lstm", epochs=12, batch_size=1024, lr=1e-3, hidden=96):
    device = torch.device("cpu")
    data = np.load(SPLIT_FILE)
    seq_data = np.load(SEQ_FILE)
    feat_names = seq_data["feature_names"].tolist() if "feature_names" in seq_data else [f"f{i}" for i in range(36)]
    # Fit scaler on train only — no leakage
    X_train_raw = data["X_train"]  # [N, W, 36]
    N,W,F = X_train_raw.shape
    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, F))
    def scale(arr): return (arr - scaler.mean_) / scaler.scale_  # manual for float32
    # Actually use scaler.transform per sample
    import sklearn.preprocessing
    def transform(arr):
        orig = arr.shape
        flat = arr.reshape(-1, F)
        scaled = scaler.transform(flat)
        return scaled.reshape(orig).astype(np.float32)
    X_train = transform(data["X_train"])
    X_val = transform(data["X_val"])
    X_test = transform(data["X_test"])
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    # Save scaler for inference
    joblib.dump({"scaler": scaler, "feature_names": feat_names}, MODEL_DIR / "scaler_forecaster.joblib")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    n_classes = 9
    horizon = y_train.shape[1]
    ModelCls = LSTMForecaster if model_type=="lstm" else GRUForecaster
    model = ModelCls(input_dim=F, hidden_dim=hidden, num_layers=2, num_classes=n_classes, horizon=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val = 0
    best_state = None
    print(f"Training {model_type} hidden={hidden} epochs={epochs} on CPU — train {len(train_ds):,} val {len(val_ds):,}")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss=0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # [B,K,C]
            loss = sum(criterion(logits[:,k], yb[:,k]) for k in range(horizon)) / horizon
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss+= loss.item()*len(xb)
        # val t+1 accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                pred = logits.argmax(dim=-1).cpu().numpy()  # [B,K]
                correct += (pred[:,0]==yb[:,0].numpy()).sum()
                total += len(yb)
        acc = correct/total
        print(f"Epoch {epoch:02d} loss={total_loss/len(train_ds):.4f} val_t+1_acc={acc:.4f}")
        if acc > best_val:
            best_val=acc; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "config": {"model_type":model_type,"hidden":hidden,"horizon":horizon,"input_dim":F,"num_classes":n_classes}, "val_acc": best_val, "feature_names": feat_names}, MODEL_DIR / f"{model_type}_forecaster.pt")
    print(f"Saved {MODEL_DIR / f'{model_type}_forecaster.pt'} best val_t+1_acc={best_val:.4f}")
    # quick test eval
    model.load_state_dict(best_state); model.eval()
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    with torch.no_grad():
        accs=[]
        for k in range(horizon):
            c=t=0
            for xb,yb in test_loader:
                pred = model(xb.to(device)).argmax(dim=-1).cpu().numpy()
                c+=(pred[:,k]==yb[:,k].numpy()).sum(); t+=len(yb)
            accs.append(c/t)
    print(f"Test acc per horizon: {[f't+{i+1}:{a:.4f}' for i,a in enumerate(accs)]}")
    return accs
