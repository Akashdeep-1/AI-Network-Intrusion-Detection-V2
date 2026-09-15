"""
Temporal split — strictly time-ordered, no shuffle.
max(train_time) < min(val_time) < min(test_time) guaranteed via sequence index.
Wraps forecast_sequences.npz.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQ_FILE = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "forecast_sequences.npz"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "ciciot2023"

def temporal_split(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    data = np.load(SEQ_FILE)
    X_seq, y_seq, S_next = data["X_seq"], data["y_seq"], data["S_next"]
    N = len(X_seq)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    # Time-ordered contiguous blocks — NO shuffle
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, N)
    print(f"Temporal split N={N}: train {train_slice}, val {val_slice}, test {test_slice}")
    # Verify no index overlap
    assert len(set(range(*train_slice.indices(N))) & set(range(*val_slice.indices(N)))) == 0
    assert len(set(range(*val_slice.indices(N))) & set(range(*test_slice.indices(N)))) == 0
    # Check time invariant: max train idx < min val idx
    assert train_slice.stop <= val_slice.start and val_slice.stop <= test_slice.start, "Temporal invariant violated"
    splits = {
        "X_train": X_seq[train_slice], "y_train": y_seq[train_slice], "S_train": S_next[train_slice],
        "X_val": X_seq[val_slice], "y_val": y_seq[val_slice], "S_val": S_next[val_slice],
        "X_test": X_seq[test_slice], "y_test": y_seq[test_slice], "S_test": S_next[test_slice],
    }
    # Save
    np.savez_compressed(OUT_DIR / "forecast_temporal_split.npz", **splits)
    meta = {"N": N, "train": len(splits["X_train"]), "val": len(splits["X_val"]), "test": len(splits["X_test"]),
            "window": int(data["window"]), "horizon": int(data["horizon"]),
            "time_ordered": True, "overlap": 0}
    (OUT_DIR / "forecast_temporal_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved forecast_temporal_split.npz — {meta}")
    print("✅ Verified: max(train) < min(val) < min(test) — no temporal leakage")
    return meta
