"""
CICIDS-2017 Loader — Dataset B temporal pipeline.
Handles:
- Official full Timestamp: "7/3/2017 15:56:34" / "2017-07-03 15:56:34" (A TRUE)
- Truncated bvk Timestamp: "56:34.2" (MM:SS.m) => reconstruct synthetic epoch (B derived, documented)
Parses robustly, sorts chronologically, validates.
"""
from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"

# Official capture dates per CIC docs (Monday=2017-07-03)
DAY_TO_DATE = {
    "monday": "2017-07-03",
    "tuesday": "2017-07-04",
    "wednesday": "2017-07-05",
    "thursday": "2017-07-06",
    "friday": "2017-07-07",
}

# Full timestamp formats to try (official)
_FULL_FORMATS = [
    "%m/%d/%Y %I:%M:%S %p",  # 7/3/2017 3:56:34 PM
    "%m/%d/%Y %H:%M:%S",     # 7/3/2017 15:56:34
    "%Y-%m-%d %H:%M:%S",     # 2017-07-03 15:56:34
    "%Y-%m-%d %H:%M:%S.%f",  # with millis
    "%d/%m/%Y %H:%M:%S",     # european fallback
    "%m/%d/%Y %H:%M",        # no seconds
]

_TRUNC_RE = re.compile(r"^\s*(\d{1,2}):(\d{1,2})(?:\.(\d+))?\s*$")

def _parse_full_timestamp(s: str) -> datetime | None:
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    # try truncated first
    m = _TRUNC_RE.match(s)
    if m and s.count(":") == 1 and len(s) < 12:
        return None  # truncated, not full
    for fmt in _FULL_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # pd fallback
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.to_pydatetime()
    except Exception:
        pass
    return None

def _parse_truncated(s: str) -> float | None:
    """Parse MM:SS.m -> total seconds within hour"""
    m = _TRUNC_RE.match(str(s).strip())
    if not m:
        return None
    mins = int(m.group(1))
    secs = int(m.group(2))
    frac = m.group(3)
    ms = int(frac)/ (10**len(frac)) if frac else 0.0
    return mins*60 + secs + ms

def load_cicids_day(csv_path: Path, day_key: str) -> pd.DataFrame:
    """Load one day CSV, parse Timestamp to epoch, return sorted DF.
    Adds columns: _ts (datetime), _epoch (float seconds), _day
    """
    df = pd.read_csv(csv_path, low_memory=False)
    # Normalize column names strip
    df.columns = [c.strip() for c in df.columns]
    # Find timestamp col (case-insensitive)
    ts_col = next((c for c in df.columns if c.lower() == "timestamp"), None)
    if ts_col is None:
        raise ValueError(f"No Timestamp column in {csv_path}, got {df.columns.tolist()[:5]}")
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError(f"No Label column in {csv_path}")

    # Try full parse
    parsed = df[ts_col].apply(_parse_full_timestamp)
    full_ok = parsed.notna().sum()
    if full_ok > len(df)*0.8:
        # A TRUE path
        df["_ts"] = parsed
        df["_epoch"] = df["_ts"].apply(lambda d: d.timestamp())
        df["_ts_quality"] = "A_TRUE"
    else:
        # B derived: truncated -> reconstruct synthetic epoch
        # Use day start + seconds_in_hour + unwrap + row_order tie-breaker
        secs = df[ts_col].apply(_parse_truncated)
        # If truncated parse also fails, fallback to row order (C weak)
        valid = secs.notna()
        if valid.sum() < len(df)*0.5:
            # No reliable time -> use row order as proxy (C weak, documented)
            base = datetime.strptime(DAY_TO_DATE[day_key], "%Y-%m-%d")
            df["_epoch"] = np.arange(len(df), dtype=float) + base.timestamp()
            df["_ts"] = pd.to_datetime(df["_epoch"], unit="s")
            df["_ts_quality"] = "C_PROXY_row_order"
        else:
            # Truncated timestamp is NOT reliable (47% out-of-order jitter, see investigation).
            # Use defensible derived ordering: per-day row order preserves capture sequence (B_DERIVED).
            # Timestamp fractional part kept as micro tie-breaker only, not as clock.
            base_date = DAY_TO_DATE.get(day_key.lower(), "2017-07-03")
            base = datetime.strptime(base_date, "%Y-%m-%d")
            base_ts = base.timestamp() + 9*3600  # 09:00 working hours start
            raw = secs.fillna(0).values.astype(float)
            # Map row order uniformly onto 8h working window (09:00-17:00) to get correct window density (480 windows/day)
            # This preserves capture order (B_DERIVED) without fake hour extrapolation; jitter preserves micro order
            base_date = DAY_TO_DATE.get(day_key.lower(), "2017-07-03")
            base = datetime.strptime(base_date, "%Y-%m-%d")
            base_ts = base.timestamp() + 9*3600
            jitter = (raw % 60) * 0.001
            # uniform spread over 8h = 28800 sec
            uniform = (np.arange(len(df), dtype=float) / max(1,len(df)-1)) * 28800.0
            df["_epoch"] = base_ts + uniform + jitter
            df["_ts"] = pd.to_datetime(df["_epoch"], unit="s")
            df["_ts_quality"] = "B_DERIVED_row_order_uniform_8h"

    df["_day"] = day_key.lower()
    # Drop rows with null epoch (should be none)
    before = len(df)
    df = df.dropna(subset=["_epoch"])
    if len(df) < before:
        print(f"WARNING: dropped {before-len(df)} rows null timestamp in {day_key}")
    df = df.sort_values("_epoch").reset_index(drop=True)
    return df

def load_all_cicids(raw_dir: Path = RAW_DIR, days: List[str] | None = None) -> pd.DataFrame:
    """Load all available days, concat chronologically, validate."""
    if days is None:
        days = ["monday","tuesday","wednesday","thursday","friday"]
    frames = []
    for d in days:
        p = raw_dir / f"{d}.csv"
        alt = raw_dir / f"{d.capitalize()}.csv"
        # also handle bencorn extracted? Try monday.csv existence
        if not p.exists():
            # try case variants
            candidates = list(raw_dir.glob(f"{d}*.csv")) + list(raw_dir.glob(f"{d.capitalize()}*.csv"))
            if candidates:
                p = candidates[0]
            else:
                print(f"Skipping {d}: not found")
                continue
        print(f"Loading {p} ...")
        df = load_cicids_day(p, d)
        print(f"  -> {len(df):,} rows, {df['_ts_quality'].iloc[0]}, epoch {df['_epoch'].min():.0f}..{df['_epoch'].max():.0f}, labels {df['Label'].nunique()}")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CICIDS CSVs found in {raw_dir}")
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("_epoch").reset_index(drop=True)
    # Validation
    nulls = full["_epoch"].isna().sum()
    if nulls:
        raise ValueError(f"Null epochs {nulls}")
    if not full["_epoch"].is_monotonic_increasing:
        # allow equal due to tie-breaker, but not decreasing after sort
        if (full["_epoch"].diff().dropna() < -1e-6).any():
            raise ValueError("Epoch not monotonic after sort")
    # Gap check: overnight gaps expected
    diffs = np.diff(full["_epoch"].values)
    large_gaps = (diffs > 4*3600).sum()  # >4h
    print(f"Loaded TOTAL {len(full):,} flows, time span {(full['_epoch'].max()-full['_epoch'].min())/3600:.1f}h, overnight gaps ~{large_gaps}")
    return full

def validate_timestamps(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": len(df),
        "quality": df["_ts_quality"].value_counts().to_dict() if "_ts_quality" in df.columns else {},
        "nulls": int(df["_epoch"].isna().sum()),
        "monotonic": bool(df["_epoch"].is_monotonic_increasing),
        "span_hours": float((df["_epoch"].max()-df["_epoch"].min())/3600) if len(df) else 0,
    }

if __name__ == "__main__":
    df = load_all_cicids()
    print(validate_timestamps(df))
    print(df["_ts"].head(3).tolist(), "->", df["_ts"].tail(3).tolist())
