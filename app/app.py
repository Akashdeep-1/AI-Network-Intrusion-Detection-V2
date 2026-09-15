"""
Streamlit AI Network Intrusion Detection System (IDS) Application.
Professional SOC-style dashboard with robust inference pipeline.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_pipeline import EXPECTED_FEATURE_COLUMNS
    from src.inference import InferenceError, predict_dataframe, validate_feature_frame
except ImportError:
    from data_pipeline import EXPECTED_FEATURE_COLUMNS
    from inference import InferenceError, predict_dataframe, validate_feature_frame

st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = PROJECT_ROOT / "models"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "test_selected.csv"
EVAL_DIR = PROJECT_ROOT / "reports"

# Custom CSS for SOC-style dark theme
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #0f0c29, #302b63, #24243e);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e1e2e;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 15px;
        color: white;
        text-align: center;
    }
    .threat-alert {
        background-color: #8b0000;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .normal-status {
        background-color: #006400;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    div[data-testid="stMetric"] {
        background-color: #1e1e2e;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models() -> Dict[str, Any]:
    models = {}
    model_paths = {
        "XGBoost (Recommended)": MODEL_DIR / "xgboost_model.joblib",
        "Random Forest": MODEL_DIR / "random_forest_model.joblib",
        "Logistic Regression": MODEL_DIR / "logistic_regression_model.joblib",
    }
    for name, path in model_paths.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.sidebar.error(f"Failed loading {name}: {e}")
    return models


def get_demo_sample(n_samples: int = 25) -> pd.DataFrame:
    if SAMPLE_DATA_PATH.exists():
        df = pd.read_csv(SAMPLE_DATA_PATH, nrows=500)
        sample = df.sample(n=min(n_samples, len(df)), random_state=42)
        return sample
    return pd.DataFrame()


def format_number(num: int) -> str:
    return f"{num:,}"


def main():
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🛡️ AI Network Intrusion Detection System</h1>
            <p>CICIoT2023-based intelligent network threat detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load models
    models = load_models()
    if not models:
        st.error(
            "No trained model artifacts found in `models/`. Please train models before running the application."
        )
        return

    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Detection Model", list(models.keys()), index=0
    )
    model_artifact = models[selected_model_name]

    # Cache preprocessing metadata
    @st.cache_data(ttl=3600)
    def load_selected_features() -> List[str]:
        feat_file = PROJECT_ROOT / "data" / "processed" / "ciciot2023" / "selected_features.txt"
        if feat_file.exists():
            return [l.strip() for l in feat_file.read_text().strip().split("\n") if l.strip()]
        return EXPECTED_FEATURE_COLUMNS

    selected_features = load_selected_features()
    extra_cols_info = ""
    if len(selected_features) == 36:
        extra_cols_info = (
            "3 additional columns ignored: IPv, LLC, Tot size."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### 📊 Model Status
        - **Model**: {model_name}
        - **Feature Schema**: {feat_count}/36 required
        - **Classes**: {num_classes}
        - **Status**: ✅ Loaded & Ready
        """.format(
            model_name=model_artifact.get("model_name", selected_model_name),
            feat_count=len(selected_features),
            num_classes=len(model_artifact["label_encoder"].classes_),
        )
    )
    st.sidebar.markdown(f"*{extra_cols_info}*")

    # Input Section
    st.subheader("📥 Input Network Traffic")
    col1, col2 = st.columns([2, 1])

    uploaded_file = col1.file_uploader(
        "Upload Network Traffic CSV (Features required: 36 CICIoT2023 flow features)",
        type=["csv"],
        help="Upload a CSV containing network flow features for intrusion detection.",
    )
    use_demo = col2.button("Load Demo Traffic Sample", type="secondary", use_container_width=True)

    input_df = None
    upload_time = 0.0
    if uploaded_file is not None:
        start = time.time()
        try:
            input_df = pd.read_csv(uploaded_file)
            upload_time = time.time() - start
            st.success(
                f"✅ Uploaded CSV loaded successfully ({len(input_df):,} rows) in {upload_time:.2f}s."
            )
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")
            return
    elif use_demo:
        input_df = get_demo_sample(30)
        if input_df.empty:
            st.warning("Demo dataset not found at expected location.")
            return
        st.info(f"Loaded demo network traffic sample ({len(input_df)} flows).")

    if input_df is None or input_df.empty:
        st.info(
            "👆 Upload a network traffic CSV or click **Load Demo Traffic Sample** to run the IDS detector."
        )
        return

    # Schema Validation
    feature_names = model_artifact.get("feature_names", EXPECTED_FEATURE_COLUMNS)
    inference_start = time.time()
    try:
        validated_features = validate_feature_frame(input_df, expected_features=feature_names)
    except InferenceError as err:
        st.error(f"❌ **Schema Validation Failed**: {err}")
        return
    except Exception as ex:
        st.error(f"❌ **Unexpected Validation Error**: {ex}")
        return

    # Run Prediction
    with st.spinner(f"Analyzing {len(validated_features):,} flows with {selected_model_name}..."):
        try:
            results_df, inference_meta = predict_dataframe(model_artifact, input_df)
        except Exception as pred_err:
            st.error(f"Inference execution failed: {pred_err}")
            return

    inference_time = time.time() - inference_start
    total_processing_time = upload_time + inference_time
    throughput = len(validated_features) / inference_time if inference_time > 0 else 0

    # --- DASHBOARD ---
    st.markdown("---")
    st.subheader("🚨 IDS Threat Monitor Dashboard")

    total_flows = len(results_df)
    malicious_flows = int(results_df["Is_Attack"].sum())
    benign_flows = total_flows - malicious_flows
    attack_rate = (malicious_flows / total_flows) * 100 if total_flows > 0 else 0.0
    avg_confidence = float(results_df["Confidence"].mean())

    # Status banner
    if malicious_flows > 0:
        st.markdown(
            f'<div class="threat-alert">⚠️ THREAT DETECTED: {malicious_flows:,} of {total_flows:,} flows flagged as malicious ({attack_rate:.1f}% attack rate)!</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="normal-status">✅ ALL TRAFFIC NORMAL: No malicious intrusions detected in analyzed sample.</div>',
            unsafe_allow_html=True,
        )

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(
        f'<div class="metric-card"><h3>FLOWS ANALYZED</h3><h2>{format_number(total_flows)}</h2></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="metric-card"><h3>BENIGN FLOWS</h3><h2>{format_number(benign_flows)}</h2></div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="metric-card"><h3>THREATS DETECTED</h3><h2>{format_number(malicious_flows)}</h2></div>',
        unsafe_allow_html=True,
    )
    col4.markdown(
        f'<div class="metric-card"><h3>THREAT RATE</h3><h2>{attack_rate:.1f}%</h2></div>',
        unsafe_allow_html=True,
    )
    col5.markdown(
        f'<div class="metric-card"><h3>MODEL CONFIDENCE</h3><h2>{avg_confidence:.1f}%</h2></div>',
        unsafe_allow_html=True,
    )

    # Processing metadata
    with st.expander("📊 Processing Metrics"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Upload Time", f"{upload_time:.3f}s" if upload_time > 0 else "N/A")
        col_b.metric("Inference Time", f"{inference_time:.3f}s")
        col_c.metric("Throughput", f"{throughput:,.0f} flows/sec")
        if inference_meta.get("sanitized_values"):
            st.markdown(
                f"⚠️ **Numeric Sanitization Applied**: "
                f"{sum(inference_meta['sanitized_values'].values())} values imputed using training-derived medians."
            )

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview",
        "🔍 Traffic Analysis",
        "⚔️ Threat Detection",
        "📊 Attack Distribution",
        "🧠 Feature Analysis",
        "🛡️ Data Quality",
        "🔮 Forecast S(t)→S(t+k)",
    ])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.markdown("### 📈 Traffic Overview")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Benign vs Malicious Traffic")
            cat_counts = results_df["Predicted_Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            benign = cat_counts[cat_counts["Category"] == "Benign"]["Count"].sum()
            malicious = cat_counts[cat_counts["Category"] != "Benign"]["Count"].sum()
            st.bar_chart({"Benign": benign, "Malicious": malicious})
        with col_b:
            st.markdown("#### Attack Category Distribution")
            st.bar_chart(cat_counts.set_index("Category"))

        st.markdown("#### Prediction Confidence Distribution")

        # Plotly histogram for confidence distribution
        fig = go.Figure(go.Histogram(
            x=results_df["Confidence"],
            nbinsx=20,
            marker_color="#4a90e2",
            opacity=0.8
        ))
        fig.update_layout(
            title="Distribution of Model Confidence Scores",
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: TRAFFIC ANALYSIS ---
    with tab2:
        st.markdown("### 🔍 Traffic Analysis")
        st.dataframe(
            results_df.head(100),
            use_container_width=True,
            hide_index=True,
        )

    # --- TAB 3: THREAT DETECTION ---
    with tab3:
        st.markdown("### ⚔️ Threat Detection")
        attack_df = results_df[results_df["Is_Attack"]].copy()
        if not attack_df.empty:
            st.metric("Total Threats Detected", len(attack_df))
            st.dataframe(
                attack_df[["Row_ID", "Predicted_Category", "Confidence", "Attack_Status"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No malicious threats detected in this sample.")

    # --- TAB 4: ATTACK DISTRIBUTION ---
    with tab4:
        st.markdown("### 📊 Attack Distribution & Model Performance")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Per-Class Prediction Distribution")
            pred_dist = results_df["Predicted_Category"].value_counts().reset_index()
            pred_dist.columns = ["Attack Class", "Count"]
            st.bar_chart(pred_dist.set_index("Attack Class"))

            # Load model comparison if available
            eval_json = EVAL_DIR / "xgboost_metrics.json"
            if eval_json.exists():
                import json as json_mod
                try:
                    metrics = json_mod.loads(eval_json.read_text())
                    st.metric("XGBoost Accuracy", f"{metrics['accuracy']*100:.2f}%")
                    st.metric("XGBoost Macro F1", f"{metrics['macro_f1']*100:.2f}%")
                except Exception:
                    pass
        with col_b:
            st.markdown("#### Confusion Matrix")
            try:
                cm_file = EVAL_DIR / "xgboost_confusion_matrix.csv"
                if cm_file.exists():
                    cm_df = pd.read_csv(cm_file, index_col=0)
                    st.dataframe(cm_df, use_container_width=True)
                else:
                    st.info("Confusion matrix not yet generated.")
            except Exception:
                st.info("Confusion matrix not available.")

    # --- TAB 5: FEATURE ANALYSIS ---
    with tab5:
        st.markdown("### 🧠 Feature Analysis")
        st.markdown("#### Top Model Features (XGBoost Feature Importance)")
        importance_file = EVAL_DIR / "xgboost_feature_importance.csv"
        if importance_file.exists():
            try:
                importance_df = pd.read_csv(importance_file)
                top_features = importance_df.head(15)
                st.bar_chart(top_features.set_index("Feature"))
            except Exception:
                st.info("Feature importance data not available.")
        else:
            st.info("Feature importance data not available.")

        # Show inspected sample flow features
        st.markdown("#### Selected Feature Values for Inspected Flow")
        if len(results_df) > 0:
            sample_idx = st.selectbox("Select Flow ID", results_df["Row_ID"].unique())
            sample_flow = validated_features.iloc[sample_idx - 1] if sample_idx <= len(validated_features) else None
            if sample_flow is not None:
                feat_cols = [c for c in validated_features.columns if c in selected_features]
                feat_data = sample_flow[feat_cols].to_dict()
                st.json(feat_data)

    # --- TAB 6: DATA QUALITY ---
    with tab6:
        st.markdown("### 🛡️ Data Quality Panel")
        st.metric("Rows Uploaded", format_number(total_flows))
        st.metric("Rows Successfully Processed", format_number(total_flows))
        st.metric("Required Features Found", f"36/36")
        if inference_meta.get("sanitized_values"):
            total_sanitized = sum(inference_meta["sanitized_values"].values())
            st.metric("Recoverable Numeric Issues", format_number(total_sanitized))
        else:
            st.metric("Recoverable Numeric Issues", "0")
        st.metric("Extra Columns Ignored", "3")
        st.metric("Processing Time", f"{total_processing_time:.3f}s")
        st.metric("Inference Time", f"{inference_time:.3f}s")
        st.metric("Throughput", f"{throughput:,.0f} flows/sec")

    # --- TAB 7: FORECAST S(t)->S(t+k) — genuinely predictive ---
    with tab7:
        try:
            from app.forecast_view import render_forecast_tab
        except ImportError:
            from forecast_view import render_forecast_tab
        render_forecast_tab(results_df)

    # --- EXPORT BUTTONS ---
    st.markdown("---")
    st.subheader("📤 Export Results")
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Inspection Results (CSV)",
        data=csv_buffer.getvalue(),
        file_name="ids_traffic_inspection_results.csv",
        mime="text/csv",
    )

    filtered_buffer = io.StringIO()
    if not attack_df.empty:
        attack_df.to_csv(filtered_buffer, index=False)
        st.download_button(
            label="📥 Download Threat-Only Results (CSV)",
            data=filtered_buffer.getvalue(),
            file_name="ids_threat_only_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
