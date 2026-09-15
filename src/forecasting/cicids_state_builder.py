"""
CICIDS State Builder — S(t) via genuine window aggregation on _epoch.
Implements STEP 3/4/6: 60s windows (configurable 30/60/120), no label leakage in input.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 21-dim schema order (must match ForecastFeatureSchema.json)
STATE_COLS = [
    "flow_count","packet_count","byte_volume","packets_per_sec","bytes_per_sec",
    "avg_packet_size","flow_duration_mean","flow_iat_mean","flow_iat_std",
    "fwd_packets_mean","bwd_packets_mean","syn_rate","rst_rate","ack_rate",
    "fin_rate","psh_rate","unique_src_ips","unique_dst_ips","unique_dst_ports",
    "tcp_ratio","udp_ratio"
]

def _safe_div(a,b): return float(a)/float(b) if b else 0.0

def build_states(
    df: pd.DataFrame,
    window_sec: int = 60,
    epoch_col: str = "_epoch",
    day_col: str = "_day",
) -> Tuple[pd.DataFrame, Dict]:
    """Aggregate df flows into S(t) per window_sec windows.
    - Does NOT interpolate across large gaps: gaps > 30min produce empty windows but not interpolated.
    - Returns states DF indexed by window start epoch, with 21 dims + _window_start, _window_end, _day, attack_ratio, dominant_label (for eval only)
    """
    if epoch_col not in df.columns:
        raise ValueError(f"Missing {epoch_col} — run cicids_loader first")
    # Determine global time bounds per day to avoid cross-day interpolation
    # Build windows sequentially within each day, then concat
    states = []
    for day, g in df.groupby(day_col):
        g = g.sort_values(epoch_col)
        start = float(g[epoch_col].min())
        end = float(g[epoch_col].max())
        # Align to window boundary
        w_start = np.floor(start / window_sec) * window_sec
        w_end = np.ceil(end / window_sec) * window_sec
        windows = np.arange(w_start, w_end, window_sec)
        # Precompute per-row numeric need
        # Ensure required columns exist with fallbacks
        # Fill numeric NaNs
        for w0 in windows:
            w1 = w0 + window_sec
            win = g[(g[epoch_col] >= w0) & (g[epoch_col] < w1)]
            if len(win)==0:
                # Keep empty window as NaNs for gap detection, but mark
                row = {c: np.nan for c in STATE_COLS}
                row.update({"_window_start": w0, "_window_end": w1, "_day": day, "flow_count": 0,
                            "attack_ratio": 0.0, "dominant_label": "BENIGN", "is_empty": True, "is_gap": True})
                states.append(row)
                continue
            # Aggregations
            flow_count = len(win)
            # packet / byte totals: use Total Fwd/Bwd packets/length if present
            fwd_pkts = win["Total Fwd Packet"].fillna(0).sum() if "Total Fwd Packet" in win.columns else 0
            bwd_pkts = win["Total Bwd packets"].fillna(0).sum() if "Total Bwd packets" in win.columns else 0
            packet_count = float(fwd_pkts + bwd_pkts)
            fwd_bytes = win["Total Length of Fwd Packet"].fillna(0).sum() if "Total Length of Fwd Packet" in win.columns else 0
            bwd_bytes = win["Total Length of Bwd Packet"].fillna(0).sum() if "Total Length of Bwd Packet" in win.columns else 0
            byte_volume = float(fwd_bytes + bwd_bytes)
            packets_per_sec = packet_count / window_sec
            bytes_per_sec = byte_volume / window_sec
            avg_packet_size = win["Average Packet Size"].mean() if "Average Packet Size" in win.columns else (byte_volume / packet_count if packet_count else 0)
            flow_duration_mean = win["Flow Duration"].mean() if "Flow Duration" in win.columns else 0
            flow_iat_mean = win["Flow IAT Mean"].mean() if "Flow IAT Mean" in win.columns else 0
            flow_iat_std = win["Flow IAT Std"].mean() if "Flow IAT Std" in win.columns else 0
            fwd_packets_mean = win["Total Fwd Packet"].mean() if "Total Fwd Packet" in win.columns else 0
            bwd_packets_mean = win["Total Bwd packets"].mean() if "Total Bwd packets" in win.columns else 0
            syn_rate = win["SYN Flag Count"].mean() if "SYN Flag Count" in win.columns else 0
            rst_rate = win["RST Flag Count"].mean() if "RST Flag Count" in win.columns else 0
            ack_rate = win["ACK Flag Count"].mean() if "ACK Flag Count" in win.columns else 0
            fin_rate = win["FIN Flag Count"].mean() if "FIN Flag Count" in win.columns else 0
            psh_rate = win["PSH Flag Count"].mean() if "PSH Flag Count" in win.columns else 0
            unique_src_ips = win["Src IP dec"].nunique() if "Src IP dec" in win.columns else 0
            unique_dst_ips = win["Dst IP dec"].nunique() if "Dst IP dec" in win.columns else 0
            unique_dst_ports = win["Dst Port"].nunique() if "Dst Port" in win.columns else 0
            # protocol ratios
            if "Protocol" in win.columns:
                proto = win["Protocol"].astype(str)
                tcp_ratio = (proto=="6").mean() if (proto=="6").any() or (proto!="6").any() else 0
                udp_ratio = (proto=="17").mean() if (proto=="17").any() or (proto!="17").any() else 0
                # fallback if numeric 6/17 not matching, use value_counts
                if tcp_ratio==0 and udp_ratio==0:
                    tcp_ratio = float((win["Protocol"]==6).mean())
                    udp_ratio = float((win["Protocol"]==17).mean())
            else:
                tcp_ratio = udp_ratio = 0
            # label overlay FOR EVAL ONLY (not input)
            labels = win["Label"].astype(str) if "Label" in win.columns else pd.Series(["BENIGN"]*len(win))
            malicious = (~labels.str.upper().isin(["BENIGN","-1","BENIGN "])).sum()
            attack_ratio = malicious / flow_count if flow_count else 0
            dominant = labels.value_counts().idxmax() if len(labels) else "BENIGN"

            row = {
                "flow_count": flow_count, "packet_count": packet_count, "byte_volume": byte_volume,
                "packets_per_sec": packets_per_sec, "bytes_per_sec": bytes_per_sec,
                "avg_packet_size": float(avg_packet_size) if pd.notna(avg_packet_size) else 0,
                "flow_duration_mean": float(flow_duration_mean) if pd.notna(flow_duration_mean) else 0,
                "flow_iat_mean": float(flow_iat_mean) if pd.notna(flow_iat_mean) else 0,
                "flow_iat_std": float(flow_iat_std) if pd.notna(flow_iat_std) else 0,
                "fwd_packets_mean": float(fwd_packets_mean) if pd.notna(fwd_packets_mean) else 0,
                "bwd_packets_mean": float(bwd_packets_mean) if pd.notna(bwd_packets_mean) else 0,
                "syn_rate": float(syn_rate) if pd.notna(syn_rate) else 0,
                "rst_rate": float(rst_rate) if pd.notna(rst_rate) else 0,
                "ack_rate": float(ack_rate) if pd.notna(ack_rate) else 0,
                "fin_rate": float(fin_rate) if pd.notna(fin_rate) else 0,
                "psh_rate": float(psh_rate) if pd.notna(psh_rate) else 0,
                "unique_src_ips": int(unique_src_ips), "unique_dst_ips": int(unique_dst_ips), "unique_dst_ports": int(unique_dst_ports),
                "tcp_ratio": float(tcp_ratio), "udp_ratio": float(udp_ratio),
                "_window_start": w0, "_window_end": w1, "_day": day,
                "attack_ratio": float(attack_ratio), "dominant_label": str(dominant),
                "is_empty": False, "is_gap": False
            }
            states.append(row)

    states_df = pd.DataFrame(states).sort_values("_window_start").reset_index(drop=True)
    # Remove overnight empty gaps where is_empty True for long stretches? Keep for gap awareness but mark
    # For training, drop empty windows (no flows) — they represent off-hours
    nonempty = states_df[~states_df["is_empty"]].copy()
    meta = {
        "window_sec": window_sec,
        "total_windows": len(states_df),
        "nonempty_windows": len(nonempty),
        "empty_windows": int(states_df["is_empty"].sum()),
        "days": df[day_col].unique().tolist() if day_col in df.columns else [],
        "state_cols": STATE_COLS,
        "span_hours": float((states_df["_window_end"].max() - states_df["_window_start"].min())/3600) if len(states_df) else 0,
    }
    return nonempty.reset_index(drop=True), meta

def states_to_arrays(states_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return S [N,21], attack_ratio [N], dominant_label_idx [N]"""
    S = states_df[STATE_COLS].fillna(0).values.astype(np.float32)
    # defensive: replace inf
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    ar = states_df["attack_ratio"].values.astype(np.float32) if "attack_ratio" in states_df.columns else np.zeros(len(states_df), dtype=np.float32)
    labels = states_df["dominant_label"].astype(str).values if "dominant_label" in states_df.columns else np.array(["BENIGN"]*len(states_df))
    return S, ar, labels, STATE_COLS

if __name__ == "__main__":
    from .cicids_loader import load_all_cicids
    df = load_all_cicids()
    for w in [30,60,120]:
        s, meta = build_states(df, window_sec=w)
        print(f"window {w}s -> {len(s)} states, {meta['empty_windows']} empty, cols {len(STATE_COLS)}, attack_ratio mean {s['attack_ratio'].mean():.3f}")
        print(s[STATE_COLS].describe().loc["mean"].to_dict())
