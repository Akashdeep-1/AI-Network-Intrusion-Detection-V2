"""
Dataset Detection and Routing Utilities
Identifies whether uploaded CSV is CICIoT2023 or CICIDS-2017 format.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Dict, List, Optional
import pandas as pd


# CICIoT2023 expected features (36)
CICIOT2023_FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP",
    "Tot sum", "Min", "Max", "AVG", "Std", "IAT", "Number", "Variance"
]

# CICIDS-2017 expected raw columns (subset of 79) - handles variations
CICIDS2017_REQUIRED_COLS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"
]

# Aliases for common column name variations in CICIDS-2017
CICIDS2017_COL_ALIASES = {
    "Destination Port": ["Dst Port", "Destination Port", "Destination.Port"],
    "Flow Duration": ["Flow Duration", "Flow.Duration"],
    "Total Fwd Packets": ["Total Fwd Packet", "Total Fwd Packets", "Total.Fwd.Packets"],
    "Total Backward Packets": ["Total Bwd packets", "Total Bwd Packets", "Total.Backward.Packets", "Total.Bwd.Packets"],
    "Label": ["Label", "Attempted Category"]
}

CICIDS2017_KEY_COLS = {"Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Label"}


class DatasetDetectionError(Exception):
    """Raised when dataset format cannot be determined or is incompatible."""


def detect_dataset_format(df: pd.DataFrame) -> Literal["ciciot2023", "cicids2017", "unknown"]:
    """
    Detect dataset format from column names.
    Returns: 'ciciot2023', 'cicids2017', or 'unknown'
    """
    cols = set(str(c).strip() for c in df.columns)
    
    # Check for CICIoT2023 signature features - key identifying columns
    ciciot_key_features = {"Header_Length", "Protocol Type", "Time_To_Live", "Rate",
                           "fin_flag_number", "syn_flag_number", "rst_flag_number", 
                           "ack_flag_number", "HTTP", "HTTPS", "TCP", "UDP"}
    ciciot_match = sum(1 for f in ciciot_key_features if f in cols)
    if ciciot_match >= 5:  # At least 5 of 12 key features
        return "ciciot2023"
    
    # Check for CICIDS-2017 signature columns (with aliases)
    def has_key_col(key: str) -> bool:
        """Check if any alias of key column exists in df."""
        # Normalize all column names to lower for comparison
        cols_lower = {c.lower() for c in cols}
        key_lower = key.lower()
        if key_lower in cols_lower:
            return True
        for alias in CICIDS2017_COL_ALIASES.get(key, []):
            if alias.lower() in cols_lower:
                return True
        return False
    
    cicids_match = sum(1 for f in CICIDS2017_KEY_COLS if has_key_col(f))
    if cicids_match >= 4:  # At least 4 of 5 key columns
        return "cicids2017"
    
    return "unknown"


def validate_ciciot2023_schema(df: pd.DataFrame) -> List[str]:
    """Validate CICIoT2033 schema, return missing features."""
    cols = set(str(c).strip() for c in df.columns)
    missing = [f for f in CICIOT2023_FEATURES if f not in cols]
    return missing


def validate_cicids2017_schema(df: pd.DataFrame) -> List[str]:
    """Validate CICIDS-2017 schema, return missing key columns."""
    cols = set(str(c).strip() for c in df.columns)
    cols_lower = {c.lower() for c in cols}
    
    missing = []
    for key in CICIDS2017_KEY_COLS:
        key_lower = key.lower()
        if key_lower in cols_lower:
            continue
        found = False
        for alias in CICIDS2017_COL_ALIASES.get(key, []):
            if alias.lower() in cols_lower:
                found = True
                break
        if not found:
            missing.append(key)
    return missing


def get_dataset_info(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get comprehensive dataset info for error messages.
    """
    detected = detect_dataset_format(df)
    cols = list(df.columns)
    
    info = {
        "detected_format": detected,
        "num_columns": len(cols),
        "column_sample": cols[:10],
    }
    
    if detected == "ciciot2023":
        missing = validate_ciciot2023_schema(df)
        info["missing_features"] = missing
        info["expected_features"] = CICIOT2023_FEATURES
    elif detected == "cicids2017":
        missing = validate_cicids2017_schema(df)
        info["missing_features"] = missing
        info["expected_key_columns"] = list(CICIDS2017_KEY_COLS)
    else:
        ciciot_missing = validate_ciciot2023_schema(df)
        cicids_missing = validate_cicids2017_schema(df)
        info["ciciot2023_missing"] = ciciot_missing
        info["cicids2017_missing"] = cicids_missing
    
    return info


def format_dataset_error(info: Dict) -> str:
    """Format user-friendly error message."""
    detected = info["detected_format"]
    
    if detected == "ciciot2023":
        missing = info.get("missing_features", [])
        if missing:
            return (
                f"❌ **CICIoT2023 Schema Incomplete**\n"
                f"Detected CICIoT2023 format but missing {len(missing)} required features:\n"
                f"{', '.join(sorted(missing))}\n\n"
                f"Expected 36 features. Ensure preprocessing pipeline has been run."
            )
        return "✅ CICIoT2023 schema validated."
    
    elif detected == "cicids2017":
        missing = info.get("missing_features", [])
        if missing:
            return (
                f"❌ **CICIDS-2017 Schema Incomplete**\n"
                f"Detected CICIDS-2017 format but missing key columns:\n"
                f"{', '.join(sorted(missing))}\n\n"
                f"Expected columns: {', '.join(info['expected_key_columns'])}"
            )
        return (
            f"⚠️ **CICIDS-2017 Data Detected — Wrong Pipeline**\n"
            f"This data uses CICIDS-2017 format (79 flow features with Timestamp, Label).\n"
            f"The **Detection tab uses CICIoT2023 model** (36 different features).\n\n"
            f"**Options:**\n"
            f"1. Use **Tab 7 (Forecasting)** for CICIDS-2017 temporal analysis\n"
            f"2. Upload CICIoT2023 data (36 features: Header_Length, Protocol Type, etc.)\n"
            f"3. Preprocess CICIDS-2017 through its own pipeline first"
        )
    
    else:
        return (
            f"❌ **Unknown Dataset Format**\n"
            f"Uploaded CSV has {info['num_columns']} columns but matches neither dataset:\n"
            f"- Sample columns: {', '.join(info['column_sample'])}\n\n"
            f"**Expected CICIoT2023 (36 features):** Header_Length, Protocol Type, Time_To_Live, Rate, fin_flag_number, ...\n"
            f"**Expected CICIDS-2017 (79 cols):** Destination Port, Flow Duration, Total Fwd Packets, Label, ...\n\n"
            f"CICIoT2023 missing: {len(info.get('ciciot2023_missing', []))} features\n"
            f"CICIDS-2017 missing: {len(info.get('cicids2017_missing', []))} key columns"
        )


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    
    # Test CICIoT2023-like
    df1 = pd.DataFrame({f: [1.0] for f in CICIOT2023_FEATURES[:5]})
    df1["Label"] = ["BENIGN"]
    print("CICIoT2023 test:", detect_dataset_format(df1))
    
    # Test CICIDS-2017-like
    df2 = pd.DataFrame({"Destination Port": [80], "Flow Duration": [100], "Total Fwd Packets": [5], "Total Backward Packets": [3], "Label": ["BENIGN"]})
    print("CICIDS-2017 test:", detect_dataset_format(df2))
    
    # Test unknown
    df3 = pd.DataFrame({"A": [1], "B": [2]})
    print("Unknown test:", detect_dataset_format(df3))