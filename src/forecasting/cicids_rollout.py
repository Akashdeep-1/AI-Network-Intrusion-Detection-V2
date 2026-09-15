"""
CICIDS Trajectory engine + rollout evaluator
STEP 9/13/14
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path

from .cicids_model import CICIDSForecaster, GRUForecaster
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

def load_model(model_path: Path, device="cpu"):
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    F = cfg["input_dim"]; K=cfg["K"]; hidden=cfg["hidden"]
    mtype=cfg["model_type"]
    ModelCls = GRUForecaster if mtype=="gru" else CICIDSForecaster
    model = ModelCls(input_dim=F, hidden_dim=hidden, num_layers=cfg["layers"], horizon=K, dropout=cfg["dropout"], mode=cfg["mode"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler_mean = np.array(ckpt["scaler_mean"]); scaler_scale=np.array(ckpt["scaler_scale"])
    # reconstruct scaler-like
    class Dummy: pass
    # we store mean/scale only; create object with transform
    return model, cfg, (scaler_mean, scaler_scale)

def rollout_predict(model, scaler_tuple, X_seq_scaled: np.ndarray, steps=5):
    """Single rollout: X_seq [W,F] scaled -> predicts next K states autoregressively?
    Our model predicts K heads in one shot (no autoregressive error accumulation).
    For true rollout K=5 we already have one-shot 5 heads.
    This function returns that.
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_seq_scaled[None,:,:]).float()  # [1,W,F]
        out = model(x)
        state = out.get("state")
        prob = torch.sigmoid(out.get("bin_logit")).cpu().numpy()[0] if "bin_logit" in out else None
        state = state.cpu().numpy()[0] if state is not None else None
    return state, prob

def format_trajectory(current_state: np.ndarray, cols: list, future_state: np.ndarray, future_prob: np.ndarray, window_sec=60):
    """Return human trajectory dict for dashboard"""
    traj=[]
    traj.append({"t":"observed S(t)", "state": dict(zip(cols, current_state.tolist())), "attack_prob": None})
    for k in range(len(future_state) if future_state is not None else len(future_prob)):
        traj.append({
            "t": f"t+{k+1} ({window_sec*(k+1)}s)",
            "state": dict(zip(cols, future_state[k].tolist())) if future_state is not None else None,
            "attack_prob": float(future_prob[k]) if future_prob is not None else None,
        })
    return traj
