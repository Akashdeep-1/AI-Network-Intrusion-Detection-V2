"""
CICIDS Train — Dataset B S(t) R^21
Implements STEP 8/12: lightweight LSTM/GRU, scaler fit on train only, early stopping
Saves model + scaler + config + metrics
"""
from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed" / "cicids2017"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

from .cicids_loader import load_all_cicids
from .cicids_state_builder import build_states, states_to_arrays
from .cicids_sequence import build_sequences, temporal_split
from .cicids_model import CICIDSForecaster, GRUForecaster
from .cicids_baselines import persistence_baseline_state, evaluate_state_mse, evaluate_bin

def prepare_data(window_sec=60, W=10, K=5, stride=1):
    print(f"Preparing data window={window_sec}s W={W} K={K}...")
    df = load_all_cicids()
    states_df, meta = build_states(df, window_sec=window_sec)
    S, ar, labs, cols = states_to_arrays(states_df)
    window_times = states_df["_window_start"].values.astype(float)
    print(f"States {S.shape}, attack_ratio {ar.mean():.3f}, labs {len(np.unique(labs))}")
    seq = build_sequences(S, ar, labs, window_times, W=W, K=K, stride=stride)
    print(f"Sequences X {seq['X_seq'].shape} S_next {seq['S_next'].shape}")
    splits = temporal_split(seq, train_ratio=0.70, val_ratio=0.15)
    # also save states_df for inspection
    PROCESS_DIR = PROCESSED
    PROCESS_DIR.mkdir(parents=True, exist_ok=True)
    states_df.to_csv(PROCESS_DIR / f"states_{window_sec}s.csv", index=False)
    print(f"Saved states to {PROCESS_DIR / f'states_{window_sec}s.csv'}")
    # Save splits
    np.savez_compressed(PROCESS_DIR / f"cicids_sequences_{window_sec}s_W{W}_K{K}.npz",
        X_train=splits["X_train"], X_val=splits["X_val"], X_test=splits["X_test"],
        S_next_train=splits["S_next_train"], S_next_val=splits["S_next_val"], S_next_test=splits["S_next_test"],
        y_ratio_train=splits["y_ratio_train"], y_ratio_val=splits["y_ratio_val"], y_ratio_test=splits["y_ratio_test"],
        y_bin_train=splits["y_bin_train"], y_bin_val=splits["y_bin_val"], y_bin_test=splits["y_bin_test"],
        y_high_train=splits["y_high_train"], y_high_val=splits["y_high_val"], y_high_test=splits["y_high_test"],
        t_start_train=splits["t_start_train"], t_start_val=splits["t_start_val"], t_start_test=splits["t_start_test"],
        cols=np.array(cols), W=np.array(W), K=np.array(K), window_sec=np.array(window_sec)
    )
    print(f"Saved sequences to cicids_sequences_{window_sec}s_W{W}_K{K}.npz")
    return splits, cols, meta


def train_forecaster(
    splits, cols, window_sec=60, W=10, K=5,
    model_type="gru", hidden=64, layers=2, dropout=0.2,
    epochs=30, batch_size=64, lr=1e-3, patience=7,
    mode="both",  # state|bin|both
    device="cpu"
):
    device = torch.device(device)
    # Scaler fit on train only
    X_train_raw = splits["X_train"]  # [N,W,21]
    F = X_train_raw.shape[2]
    scaler = StandardScaler()
    flat = X_train_raw.reshape(-1, F)
    scaler.fit(flat)
    def transform(arr):
        shp = arr.shape
        flat = arr.reshape(-1, F)
        scaled = scaler.transform(flat)
        return scaled.reshape(shp).astype(np.float32)
    X_train = transform(splits["X_train"])
    X_val = transform(splits["X_val"])
    X_test = transform(splits["X_test"])
    S_train = transform(splits["S_next_train"]) if mode in ("state","both") else None
    S_val = transform(splits["S_next_val"]) if mode in ("state","both") else None
    S_test = transform(splits["S_next_test"]) if mode in ("state","both") else None

    y_bin_train = splits["y_bin_train"]; y_bin_val = splits["y_bin_val"]; y_bin_test = splits["y_bin_test"]
    # Save scaler
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "cols": cols, "window_sec": window_sec, "W": W, "K": K}, MODEL_DIR / f"cicids_scaler_W{W}_K{K}.joblib")
    print(f"Scaler fitted on train {X_train.shape}, mean {scaler.mean_[:3].round(2).tolist()}...")

    # Datasets
    # For 'both', we train two losses jointly
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_bin_train) if mode in ("bin","both") else torch.zeros(len(X_train)), torch.from_numpy(S_train) if S_train is not None else torch.zeros(len(X_train),K,F))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_bin_val) if mode in ("bin","both") else torch.zeros(len(X_val)), torch.from_numpy(S_val) if S_val is not None else torch.zeros(len(X_val),K,F))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    ModelCls = GRUForecaster if model_type=="gru" else CICIDSForecaster
    model = ModelCls(input_dim=F, hidden_dim=hidden, num_layers=layers, horizon=K, dropout=dropout, mode=mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # losses
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ce_high = nn.CrossEntropyLoss()

    best_val = float("inf") if mode in ("state","both") else -1
    best_state = None
    patience_cnt = 0
    start = time.time()
    print(f"Training {model_type} hidden={hidden} layers={layers} mode={mode} epochs={epochs} train={len(train_ds)} val={len(val_ds)}")

    for epoch in range(1, epochs+1):
        model.train()
        tot_loss=0
        for xb, yb_b, yb_s in train_loader:
            xb = xb.to(device)
            out = model(xb)
            loss = 0
            if mode in ("state","both"):
                # yb_s is S_next already transformed
                y_true = yb_s.to(device)  # [B,K,F]
                pred = out["state"]  # [B,K,F]
                loss_state = mse(pred, y_true)
                loss = loss + loss_state
            if mode in ("bin","both"):
                y_true_b = yb_b.float().to(device)  # [B,K]
                pred_b = out["bin_logit"]  # [B,K]
                loss_bin = bce(pred_b, y_true_b)
                loss = loss + loss_bin if mode=="both" else loss_bin
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item()*len(xb)
        avg_loss = tot_loss/len(train_ds)
        # val
        model.eval()
        with torch.no_grad():
            v_loss=0; correct=0; total=0
            for xb, yb_b, yb_s in val_loader:
                xb = xb.to(device)
                out = model(xb)
                vl=0
                if mode in ("state","both"):
                    vl += mse(out["state"], yb_s.to(device)).item()
                if mode in ("bin","both"):
                    vl += bce(out["bin_logit"], yb_b.float().to(device)).item()
                v_loss += vl*len(xb)
                if mode in ("bin","both"):
                    preds = (torch.sigmoid(out["bin_logit"]) > 0.5).cpu().numpy()
                    correct += (preds == yb_b.numpy()).sum()
                    total += yb_b.numel()
            v_loss /= len(val_ds)
            acc = correct/total if total else 0
        print(f"Epoch {epoch:02d} train_loss={avg_loss:.4f} val_loss={v_loss:.4f}" + (f" val_bin_acc={acc:.4f}" if mode in ("bin","both") else f" v_state_mse={v_loss:.4f}"))
        # early stopping on val_loss (for state) or -acc
        improved=False
        if mode in ("state","both"):
            if v_loss < best_val - 1e-4:
                best_val=v_loss; improved=True
        else:
            if acc > best_val:
                best_val=acc; improved=True
        if improved:
            best_state={k:v.cpu() for k,v in model.state_dict().items()}
            patience_cnt=0
        else:
            patience_cnt+=1
            if patience_cnt>=patience:
                print(f"Early stop at epoch {epoch} best {best_val:.4f}")
                break
    elapsed = time.time()-start
    print(f"Training done {elapsed:.1f}s best_val {best_val:.4f}")
    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    # save
    torch.save({
        "model_state": best_state,
        "config": {"model_type": model_type, "hidden": hidden, "layers": layers, "dropout": dropout, "mode": mode, "window_sec": window_sec, "W": W, "K": K, "input_dim": F, "epochs": epochs},
        "val_metric": float(best_val),
        "cols": cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }, MODEL_DIR / f"cicids_{model_type}_W{W}_K{K}_{window_sec}s.pt")
    print(f"Saved model to {MODEL_DIR / f'cicids_{model_type}_W{W}_K{K}_{window_sec}s.pt'}")
    # evaluate on test
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_bin_test), torch.from_numpy(S_test) if S_test is not None else torch.zeros(len(X_test),K,F))
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    model.eval()
    state_preds=[]; state_trues=[]
    bin_preds=[]; bin_trues=[]; bin_probs=[]
    with torch.no_grad():
        for xb, yb_b, yb_s in test_loader:
            out = model(xb.to(device))
            if mode in ("state","both"):
                state_preds.append(out["state"].cpu().numpy())
                state_trues.append(yb_s.numpy())
            if mode in ("bin","both"):
                prob = torch.sigmoid(out["bin_logit"]).cpu().numpy()
                bin_probs.append(prob)
                bin_preds.append((prob>0.5).astype(np.int64))
                bin_trues.append(yb_b.numpy())
    results={}
    if mode in ("state","both"):
        state_preds = np.concatenate(state_preds, axis=0)
        state_trues = np.concatenate(state_trues, axis=0)
        # inv transform? evaluate in scaled space + also original? For now scaled MSE (like standard)
        # baselines
        pers = persistence_baseline_state(S_test, X_test)
        mse_pers = evaluate_state_mse(pers, S_test)
        mse_model = evaluate_state_mse(state_preds, state_trues)
        print(f"Test STATE MSE model {mse_model['mse_mean']:.6f} vs persistence {mse_pers['mse_mean']:.6f} delta {mse_pers['mse_mean']-mse_model['mse_mean']:.6f}")
        print(f"  per K model { [round(v,5) for v in mse_model['mse_per_k']]}")
        print(f"  per K pers  { [round(v,5) for v in mse_pers['mse_per_k']]}")
        results["state"]={"model": mse_model, "persistence": mse_pers, "delta_mse": float(mse_pers["mse_mean"]-mse_model["mse_mean"])}
    if mode in ("bin","both"):
        bin_preds = np.concatenate(bin_preds, axis=0)
        bin_trues = np.concatenate(bin_trues, axis=0)
        bin_probs = np.concatenate(bin_probs, axis=0)
        eval_bin = evaluate_bin(bin_preds, bin_trues, bin_probs)
        # majority baseline
        maj = int(np.bincount(splits["y_bin_train"].reshape(-1)).argmax())
        maj_pred = np.full_like(bin_trues, maj)
        eval_maj = evaluate_bin(maj_pred, bin_trues)
        print(f"Test BIN flat_acc model {eval_bin['flat_acc']:.4f} f1 {eval_bin['flat_f1']:.4f} ROC {eval_bin.get('roc_auc',0):.4f} vs majority acc {eval_maj['flat_acc']:.4f}")
        for k in range(K):
            print(f"  t+{k+1} acc {eval_bin[f't+{k+1}']['acc']:.4f} f1 {eval_bin[f't+{k+1}']['f1']:.4f}")
        results["bin"]={"model": eval_bin, "majority": eval_maj}
    # save reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / f"cicids_forecast_metrics_W{W}_K{K}_{window_sec}s.json","w") as f:
        json.dump({"config": {"window_sec": window_sec, "W": W, "K": K, "model_type": model_type, "hidden": hidden, "mode": mode}, "results": results, "best_val": float(best_val), "elapsed": elapsed}, f, indent=2)
    # decay csv
    if "state" in results:
        import csv
        with open(REPORT_DIR / f"cicids_forecast_decay_W{W}_K{K}_{window_sec}s.csv","w", newline="") as f:
            w=csv.writer(f); w.writerow(["horizon","model_mse","persistence_mse","model_mae","persistence_mae"])
            for k in range(K):
                w.writerow([f"t+{k+1}", results["state"]["model"]["mse_per_k"][k], results["state"]["persistence"]["mse_per_k"][k], results["state"]["model"]["mae_per_k"][k], results["state"]["persistence"]["mae_per_k"][k]])
    print("Reports saved to", REPORT_DIR)
    return results

if __name__ == "__main__":
    splits, cols, meta = prepare_data(window_sec=60, W=10, K=5)
    train_forecaster(splits, cols, window_sec=60, W=10, K=5, model_type="gru", hidden=64, mode="both", epochs=30)
