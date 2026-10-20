"""
CerebroTwin - Personalized Digital Twin Dashboard
Forecasting Orthostatic Cerebral Blood-Flow-Velocity Instability in Type 2 Diabetes

CRITICAL DOMAIN RULES:
1. ALWAYS use "MCA blood-flow velocity" or "MCAv" (cm/s), NEVER "cerebral blood flow" or volumetric perfusion.
2. MCAv (TCD) is GROUND TRUTH for training ONLY — NEVER a runtime input feature.
3. Abnormality flags: "Research Response Status: HIGH DEVIATION" — NEVER clinical diagnosis.
4. t=0 is the sit-to-stand transition moment. Waveforms span -60s to +60s.
5. Subject-level splitting only (GroupKFold / LeaveOneGroupOut by subject_id).
6. Cohort structure: N=39 subjects (20 T2D, 19 Controls, aged 50-85).
"""

import sys
from pathlib import Path

# Add project root to sys.path first before any module imports
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
project_root = ROOT_DIR

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml
import joblib

# CerebroTwin Core Modules
from src.data.loader import CerebroTwinDataLoader
from src.data.preprocessing import WaveformPreprocessor
from src.features.digital_twin_state import DigitalTwinState
from src.features.feature_engineering import (
    extract_features_for_cohort,
    extract_targets_for_cohort,
    get_feature_names,
    get_target_names,
    get_trajectory_target_names,
)
from src.models.baselines import MeanBaselineModel, RidgeBaselineModel
from src.models.counterfactuals import WhatIfSimulator
from src.models.evaluator import ModelEvaluator
from src.models.main_xgboost import XGBoostTwinModel

# -----------------------------------------------------------------------------
# Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CerebroTwin | Digital Twin for Orthostatic MCAv",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Standardized Color Palette
COLOR_T2D = "#E63946"        # Crimson Red
COLOR_CONTROL = "#457B9D"    # Slate Teal / Blue
COLOR_NEUTRAL = "#1D3557"    # Deep Navy
COLOR_ACCENT = "#2A9D8F"     # Emerald Green
COLOR_WARNING = "#F4A261"    # Amber
COLOR_BG_CARD = "#F8F9FA"

# Custom CSS for Sleek Scientific Look
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1d3557;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    .badge-t2d {
        background-color: #ffebee;
        color: #c62828;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-control {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-box-normal {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 14px;
        border-radius: 4px;
        margin: 10px 0px;
    }
    .status-box-deviation {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
        padding: 14px;
        border-radius: 4px;
        margin: 10px 0px;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #795548;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_and_preprocess_cohort(
    _config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Load raw/synthetic cohort, apply preprocessing, and extract standardized features and targets."""
    loader = CerebroTwinDataLoader(_config)
    subjects_df, raw_waveforms = loader.load()

    preprocessor = WaveformPreprocessor(_config)
    processed_waveforms = preprocessor.process(raw_waveforms)

    features_df = extract_features_for_cohort(subjects_df, processed_waveforms, _config)
    targets_df = extract_targets_for_cohort(subjects_df, processed_waveforms, _config)

    return subjects_df, raw_waveforms, processed_waveforms, features_df, targets_df


@st.cache_resource(show_spinner=False)
def train_cohort_models(
    features_df: pd.DataFrame, targets_df: pd.DataFrame, _config: Dict[str, Any]
) -> Dict[str, Any]:
    """Train Mean, Ridge, and XGBoost Digital Twin models or load persisted ones if available."""
    models_dir = project_root / "models"
    xgb_path = models_dir / "xgboost_twin.joblib"
    ridge_path = models_dir / "ridge_baseline.joblib"
    mean_path = models_dir / "mean_baseline.joblib"

    if xgb_path.exists() and ridge_path.exists() and mean_path.exists():
        try:
            mean_model = joblib.load(mean_path)
            ridge_model = joblib.load(ridge_path)
            xgb_model = joblib.load(xgb_path)
            return {
                "mean": mean_model,
                "ridge": ridge_model,
                "xgboost": xgb_model,
            }
        except Exception:
            pass

    # If missing on disk, train from scratch and persist
    models_dir.mkdir(parents=True, exist_ok=True)
    mean_model = MeanBaselineModel()
    mean_model.fit(features_df, targets_df)
    joblib.dump(mean_model, mean_path)

    ridge_alpha = float(_config.get("model", {}).get("ridge_alpha", 10.0))
    ridge_model = RidgeBaselineModel(alpha=ridge_alpha)
    ridge_model.fit(features_df, targets_df)
    joblib.dump(ridge_model, ridge_path)

    xgb_model = XGBoostTwinModel(_config)
    xgb_model.fit(features_df, targets_df)
    joblib.dump(xgb_model, xgb_path)

    joblib.dump(get_feature_names(), models_dir / "feature_names.joblib")
    joblib.dump(get_target_names() + get_trajectory_target_names(), models_dir / "target_names.joblib")

    return {
        "mean": mean_model,
        "ridge": ridge_model,
        "xgboost": xgb_model,
    }


@st.cache_data(show_spinner=False)
def run_cached_evaluation(
    features_df: pd.DataFrame, targets_df: pd.DataFrame, _config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run GroupKFold cross-validation across all 3 models or load persisted results."""
    eval_joblib_path = ROOT_DIR / "models" / "eval_dict.joblib"
    eval_csv_path = ROOT_DIR / "models" / "evaluation_results.csv"

    if eval_joblib_path.exists():
        try:
            return joblib.load(eval_joblib_path)
        except Exception:
            pass

    evaluator = ModelEvaluator(_config)
    groups = features_df.index.to_numpy()

    mean_model = MeanBaselineModel()
    ridge_alpha = float(_config.get("model", {}).get("ridge_alpha", 10.0))
    ridge_model = RidgeBaselineModel(alpha=ridge_alpha)
    xgb_model = XGBoostTwinModel(_config)

    mean_res = evaluator.evaluate_group_kfold(mean_model, features_df, targets_df, groups=groups, n_splits=5)
    ridge_res = evaluator.evaluate_group_kfold(ridge_model, features_df, targets_df, groups=groups, n_splits=5)
    xgb_res = evaluator.evaluate_group_kfold(xgb_model, features_df, targets_df, groups=groups, n_splits=5)

    comparison_df = evaluator.compare_models([mean_res, ridge_res, xgb_res])

    res = {
        "mean_results": mean_res,
        "ridge_results": ridge_res,
        "xgb_results": xgb_res,
        "comparison_df": comparison_df,
    }
    try:
        eval_joblib_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(res, eval_joblib_path)
        comparison_df.to_csv(eval_csv_path, index=False)
    except Exception:
        pass

    return res


def load_application_config() -> Dict[str, Any]:
    """Safely load config.yaml from project root."""
    config_path = project_root / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Default fallback config if file is inaccessible
    return {
        "data": {
            "mode": "synthetic",
            "n_t2d": 20,
            "n_control": 19,
            "sampling_rate_hz": 50,
            "window_pre_stand_s": 60,
            "window_post_stand_s": 60,
        },
        "model": {
            "n_trajectory_knots": 13,
            "trajectory_times": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            "quantiles": [0.05, 0.5, 0.95],
            "random_state": 42,
            "ridge_alpha": 10.0,
            "xgb_params": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.05,
                "reg_alpha": 1.5,
                "reg_lambda": 5.0,
            },
        },
    }


def ensure_auto_bootstrap(_cfg: Dict[str, Any]) -> None:
    """Verify pre-trained model artifacts, evaluation benchmarks, and cohort data exist on disk."""
    models_dir = ROOT_DIR / "models"
    data_dir = ROOT_DIR / "data" / "synthetic"
    xgb_path = models_dir / "xgboost_twin.joblib"
    eval_csv_path = models_dir / "evaluation_results.csv"
    meta_path = data_dir / "metadata.csv"

    needs_bootstrap = (
        not xgb_path.exists()
        or not eval_csv_path.exists()
        or not meta_path.exists()
    )
    if needs_bootstrap:
        with st.spinner("Initializing CerebroTwin Digital Twin & Training Models..."):
            from main import cmd_generate, cmd_train, cmd_evaluate
            s_df, wfs = cmd_generate(_cfg)
            cmd_train(_cfg, subjects_df=s_df, waveforms=wfs)
            cmd_evaluate(_cfg, subjects_df=s_df, waveforms=wfs)


# =============================================================================
# Main Application Initialization
# =============================================================================
config = load_application_config()
ensure_auto_bootstrap(config)

with st.spinner("Initializing CerebroTwin Digital Twin Engine..."):
    subjects_df, raw_wf, processed_wf, features_df, targets_df = load_and_preprocess_cohort(config)
    models_dict = train_cohort_models(features_df, targets_df, config)

xgb_model: XGBoostTwinModel = models_dict["xgboost"]
ridge_model: RidgeBaselineModel = models_dict["ridge"]
mean_model: MeanBaselineModel = models_dict["mean"]

# Calculate Control Group 90th percentile threshold for Research Response Status
control_sids = subjects_df[subjects_df["group"].astype(str).str.upper() == "CONTROL"]["subject_id"].tolist()
if not control_sids:
    control_sids = subjects_df[subjects_df["t2d_status"] == 0]["subject_id"].tolist()

control_targets = targets_df.loc[targets_df.index.intersection(control_sids)]
control_drop_mean = float(control_targets["pct_mca_drop"].mean())
control_drop_std = float(control_targets["pct_mca_drop"].std())
control_drop_90th = float(np.percentile(control_targets["pct_mca_drop"].dropna(), 90))
z_score_threshold_val = control_drop_mean + 1.65 * control_drop_std
effective_threshold = min(control_drop_90th, z_score_threshold_val)

# Sidebar: Global Navigation & Subject Selector
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=64)
st.sidebar.title("CerebroTwin")
st.sidebar.caption("Personalized Digital Twin for Forecasting Orthostatic MCAv Instability")

st.sidebar.markdown("---")
selected_tab = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Overview",
        "📊 Cohort Explorer",
        "🌊 Waveform Viewer",
        "🧬 Digital Twin Prediction",
        "📈 Model Evaluation",
        "🔮 What-If Simulator",
        "📋 Research Report",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Active Cohort Filter")
cohort_mode = config.get("data", {}).get("mode", "synthetic").title()
st.sidebar.info(f"**Data Pipeline:** {cohort_mode} Engine\n\n**Subjects:** N=39 (20 T2D, 19 Controls)\n**Telemetry:** -60s to +60s @ 50 Hz")

# Helper: format subject dropdown option
subject_list = list(subjects_df["subject_id"].values)
subject_display_names = {}
for _, row in subjects_df.iterrows():
    sid = str(row["subject_id"])
    grp = str(row["group"]).upper()
    age = int(row.get("age", 65))
    sex = str(row.get("sex", "M")).upper()
    hba1c = float(row.get("hba1c", 5.5))
    subject_display_names[sid] = f"{sid} | {grp} | {sex}, {age}y | HbA1c {hba1c:.1f}%"


# =============================================================================
# TAB 1: 🏠 OVERVIEW
# =============================================================================
if selected_tab == "🏠 Overview":
    st.title("🧠 CerebroTwin Digital Twin Platform")
    st.markdown(
        """
        ### Forecasting Orthostatic Cerebral Blood-Flow-Velocity Instability in Type 2 Diabetes
        **CerebroTwin** is a personalized physiological in-silico twin architecture that predicts transient 
        orthostatic **MCA blood-flow velocity (MCAv)** instability during sit-to-stand transitions in patients 
        with Type 2 Diabetes (T2D).
        """
    )

    st.markdown("---")

    # High-level Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Total Study Cohort</div>
                <div class="metric-value">39 Subjects</div>
                <div class="metric-delta">20 T2D (51.3%) | 19 Controls (48.7%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Orthostatic Protocol</div>
                <div class="metric-value">120 Seconds</div>
                <div class="metric-delta">-60s Seated Baseline ➔ +60s Active Stand</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Signal Ingestion</div>
                <div class="metric-value">50 Hz Continuous</div>
                <div class="metric-delta">ECG HR, SBP, DBP, MAP, EtCO2, RespRate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Target Biomarker</div>
                <div class="metric-value">MCAv (cm/s)</div>
                <div class="metric-delta">TCD Ground Truth Only (Zero Leakage)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Core System Architecture & Clinical Challenge
    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.subheader("Clinical Challenge: Dynamic Cerebral Autoregulation (dCA)")
        st.markdown(
            """
            Patients with **Type 2 Diabetes (T2D)** exhibit marked microvascular stiffening, autonomic neuropathy, 
            and endothelial dysfunction that blunt the rapid myogenic and metabolic vasodilation of cerebral arterioles. 
            Upon transitioning from sitting to standing (t=0):
            - **Healthy Controls:** Baroreceptors promptly trigger systemic vasoconstriction and tachycardia, while intact 
              cerebral autoregulation dilates middle cerebral artery (MCA) downstream arterioles, limiting MCA blood-flow velocity 
              drops to 5-18% and restoring baseline velocity within 10-15 seconds.
            - **T2D Patients:** Impaired baroreflex and delayed myogenic compensation result in severe, prolonged 
              drops in MCA blood-flow velocity (>25-45%), causing cerebral hypoperfusion, presyncope, and falls.
            
            **The CerebroTwin Solution:** Continuous Transcranial Doppler (TCD) ultrasonography requires specialized sonographers 
            and bulky headgear. CerebroTwin acts as a non-invasive software sensor: it calibrates exclusively on peripheral vitals 
            (continuous arterial pressure, ECG, end-tidal capnography, and clinical history) to forecast the 0-60s post-stand 
            MCAv trajectory and quantify acute hypoperfusion vulnerability.
            """
        )

    with c_right:
        st.subheader("Cohort Distribution Summary")
        grp_counts = subjects_df["group"].value_counts().reset_index()
        grp_counts.columns = ["Cohort", "Count"]
        fig_pie = px.pie(
            grp_counts,
            names="Cohort",
            values="Count",
            color="Cohort",
            color_discrete_map={"T2D": COLOR_T2D, "Control": COLOR_CONTROL},
            hole=0.45,
        )
        fig_pie.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Architectural Principles Callout Box
    st.info(
        """
        **Strict Digital Twin Scientific Rules:**
        1. **MCA blood-flow velocity (MCAv in cm/s)** is strictly Ground Truth for offline supervised training. It is **never** provided as a runtime input.
        2. **Subject-level Data Isolation:** Cross-validation is partitioned strictly via `GroupKFold` on `subject_id` to prevent intra-subject telemetry leakage.
        3. **Abnormality Flagging:** Predicted anomalies are classified as `"Research Response Status: HIGH DEVIATION"` — never as clinical diagnoses.
        """
    )


# =============================================================================
# TAB 2: 📊 COHORT EXPLORER
# =============================================================================
elif selected_tab == "📊 Cohort Explorer":
    st.title("📊 Clinical Cohort Explorer (N=39)")
    st.markdown("Explore demographic, clinical, and resting autonomic profiles across T2D and Control participants.")

    # Filter Section
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        grp_filter = st.selectbox("Filter by Group", ["All Cohorts", "T2D", "Control"])
    with f2:
        min_age, max_age = int(subjects_df["age"].min()), int(subjects_df["age"].max())
        age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    with f3:
        sex_filter = st.multiselect("Biological Sex", ["M", "F"], default=["M", "F"])
    with f4:
        htn_filter = st.selectbox("Hypertension Status", ["All", "Hypertensive", "Normotensive"])

    # Apply Filters
    filtered_df = subjects_df.copy()
    if grp_filter != "All Cohorts":
        filtered_df = filtered_df[filtered_df["group"].astype(str).str.upper() == grp_filter.upper()]

    filtered_df = filtered_df[
        (filtered_df["age"] >= age_range[0])
        & (filtered_df["age"] <= age_range[1])
        & (filtered_df["sex"].isin(sex_filter))
    ]

    if htn_filter == "Hypertensive":
        filtered_df = filtered_df[filtered_df["hypertension"] == True]
    elif htn_filter == "Normotensive":
        filtered_df = filtered_df[filtered_df["hypertension"] == False]

    st.markdown(f"**Showing {len(filtered_df)} of {len(subjects_df)} subjects**")

    # Table Presentation
    display_cols = [
        "subject_id",
        "group",
        "age",
        "sex",
        "hba1c",
        "bmi",
        "resting_sbp",
        "resting_dbp",
        "resting_map",
        "resting_hr",
        "baseline_rmssd",
        "hypertension",
    ]
    present_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[present_cols].style.format(
            {
                "hba1c": "{:.1f}%",
                "bmi": "{:.1f}",
                "resting_sbp": "{:.0f}",
                "resting_dbp": "{:.0f}",
                "resting_map": "{:.0f}",
                "resting_hr": "{:.0f}",
                "baseline_rmssd": "{:.1f} ms",
            }
        ),
        use_container_width=True,
        height=280,
    )

    st.markdown("---")
    st.subheader("Biomarker Distributions: T2D vs Controls")

    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        # HbA1c Distribution with 6.5% Diagnostic Cutoff
        fig_hba1c = px.histogram(
            subjects_df,
            x="hba1c",
            color="group",
            barmode="overlay",
            nbins=15,
            color_discrete_map={"T2D": COLOR_T2D, "Control": COLOR_CONTROL},
            labels={"hba1c": "Glycated Hemoglobin HbA1c (%)", "count": "Subject Count"},
            title="HbA1c Distribution (Red Dotted: 6.5% Diagnostic Threshold)",
        )
        fig_hba1c.add_vline(x=6.5, line_dash="dash", line_color="#D90429", annotation_text="T2D Cutoff (6.5%)")
        fig_hba1c.update_layout(template="plotly_white", height=320, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_hba1c, use_container_width=True)

    with col_dist2:
        # Autonomic Vagal Tone (RMSSD ms)
        fig_rmssd = px.box(
            subjects_df,
            x="group",
            y="baseline_rmssd",
            color="group",
            points="all",
            color_discrete_map={"T2D": COLOR_T2D, "Control": COLOR_CONTROL},
            labels={"baseline_rmssd": "Baseline RMSSD (ms)", "group": "Cohort"},
            title="Parasympathetic Vagal Tone (Resting RMSSD ms)",
        )
        fig_rmssd.update_layout(template="plotly_white", height=320, margin=dict(l=40, r=20, t=40, b=40), showlegend=False)
        st.plotly_chart(fig_rmssd, use_container_width=True)

    col_dist3, col_dist4 = st.columns(2)
    with col_dist3:
        # Resting SBP Comparison
        fig_sbp = px.box(
            subjects_df,
            x="group",
            y="resting_sbp",
            color="group",
            points="all",
            color_discrete_map={"T2D": COLOR_T2D, "Control": COLOR_CONTROL},
            labels={"resting_sbp": "Resting Systolic BP (mmHg)", "group": "Cohort"},
            title="Baseline Systolic Blood Pressure Comparison",
        )
        fig_sbp.update_layout(template="plotly_white", height=320, margin=dict(l=40, r=20, t=40, b=40), showlegend=False)
        st.plotly_chart(fig_sbp, use_container_width=True)

    with col_dist4:
        # Age vs BMI Scatter
        fig_bmi = px.scatter(
            subjects_df,
            x="age",
            y="bmi",
            color="group",
            size="hba1c",
            hover_name="subject_id",
            color_discrete_map={"T2D": COLOR_T2D, "Control": COLOR_CONTROL},
            labels={"age": "Age (years)", "bmi": "BMI (kg/m²)", "hba1c": "HbA1c (%)"},
            title="Demographic Profile: Age vs BMI (Size = HbA1c)",
        )
        fig_bmi.update_layout(template="plotly_white", height=320, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_bmi, use_container_width=True)


# =============================================================================
# TAB 3: 🌊 WAVEFORM VIEWER
# =============================================================================
elif selected_tab == "🌊 Waveform Viewer":
    st.title("🌊 Multimodal Waveform Telemetry")
    st.markdown("High-resolution 50Hz continuous recordings during the 120-second sit-to-stand challenge (-60s to +60s).")

    # Subject Selector
    selected_sid = st.selectbox(
        "Select Subject to Inspect Waveforms",
        options=subject_list,
        format_func=lambda s: subject_display_names.get(s, s),
        index=0,
    )

    subj_meta = subjects_df[subjects_df["subject_id"] == selected_sid].iloc[0]
    wf = processed_wf.get(selected_sid)

    if wf is None or len(wf) == 0:
        st.error(f"No continuous waveform data found for {selected_sid}")
    else:
        # Subject Demographic Badge
        g_badge = "badge-t2d" if str(subj_meta["group"]).upper() == "T2D" else "badge-control"
        st.markdown(
            f"""
            <div style="background-color: #f1f3f5; padding: 12px 18px; border-radius: 8px; margin-bottom: 15px;">
                <span class="{g_badge}">{str(subj_meta["group"]).upper()}</span> &nbsp;
                <b>Subject ID:</b> {selected_sid} &nbsp;|&nbsp;
                <b>Age:</b> {int(subj_meta["age"])} yrs &nbsp;|&nbsp;
                <b>Sex:</b> {subj_meta["sex"]} &nbsp;|&nbsp;
                <b>BMI:</b> {float(subj_meta["bmi"]):.1f} &nbsp;|&nbsp;
                <b>HbA1c:</b> {float(subj_meta["hba1c"]):.1f}% &nbsp;|&nbsp;
                <b>Hypertension:</b> {"Yes" if subj_meta.get("hypertension", False) else "No"}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Detect MCAv Nadir in Post-Stand Window (t > 0)
        post_mask = wf["time_s"] >= 0.0
        post_wf = wf[post_mask]
        
        pre_mask = (wf["time_s"] >= -60.0) & (wf["time_s"] <= 0.0)
        baseline_mcav = float(wf.loc[pre_mask, "mcav_cms"].mean()) if "mcav_cms" in wf.columns else 60.0

        nadir_mcav = float(post_wf["mcav_cms"].min())
        nadir_row = post_wf[post_wf["mcav_cms"] == nadir_mcav].iloc[0]
        nadir_time = float(nadir_row["time_s"])
        drop_pct = float((baseline_mcav - nadir_mcav) / max(baseline_mcav, 1e-3) * 100.0)

        # Construct 5-panel synchronous subplots
        fig_wf = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=[
                "1. MCA Blood-Flow Velocity (MCAv Ground Truth, cm/s)",
                "2. Arterial Blood Pressure (SBP, DBP, MAP, mmHg)",
                "3. Heart Rate (ECG HR, bpm)",
                "4. End-Tidal Carbon Dioxide (EtCO2, mmHg)",
                "5. Respiratory Rate (breaths/min)",
            ],
        )

        t_arr = wf["time_s"]

        # Panel 1: MCAv
        fig_wf.add_trace(
            go.Scatter(
                x=t_arr,
                y=wf["mcav_cms"],
                mode="lines",
                name="MCAv (cm/s)",
                line=dict(color="#9D0208" if str(subj_meta["group"]).upper() == "T2D" else "#1D3557", width=2.0),
            ),
            row=1,
            col=1,
        )
        # Mark MCAv Nadir
        fig_wf.add_trace(
            go.Scatter(
                x=[nadir_time],
                y=[nadir_mcav],
                mode="markers+text",
                name="MCAv Nadir",
                marker=dict(symbol="star", size=14, color="#E63946", line=dict(width=1.5, color="#000")),
                text=[f"Nadir: {nadir_mcav:.1f} cm/s (-{drop_pct:.1f}%)"],
                textposition="bottom center",
            ),
            row=1,
            col=1,
        )

        # Panel 2: SBP, DBP, MAP
        fig_wf.add_trace(
            go.Scatter(x=t_arr, y=wf["sbp_mmhg"], mode="lines", name="SBP", line=dict(color="#E63946", width=1.5)),
            row=2,
            col=1,
        )
        fig_wf.add_trace(
            go.Scatter(x=t_arr, y=wf["dbp_mmhg"], mode="lines", name="DBP", line=dict(color="#457B9D", width=1.5)),
            row=2,
            col=1,
        )
        fig_wf.add_trace(
            go.Scatter(x=t_arr, y=wf["map_mmhg"], mode="lines", name="MAP", line=dict(color="#1D3557", width=2.2, dash="dot")),
            row=2,
            col=1,
        )

        # Panel 3: Heart Rate
        fig_wf.add_trace(
            go.Scatter(x=t_arr, y=wf["ecg_hr_bpm"], mode="lines", name="Heart Rate", line=dict(color="#2A9D8F", width=2.0)),
            row=3,
            col=1,
        )

        # Panel 4: EtCO2
        fig_wf.add_trace(
            go.Scatter(x=t_arr, y=wf["etco2_mmhg"], mode="lines", name="EtCO2", line=dict(color="#E76F51", width=1.8)),
            row=4,
            col=1,
        )

        # Panel 5: Respiratory Rate
        fig_wf.add_trace(
            go.Scatter(x=t_arr, y=wf["resp_rate_bpm"], mode="lines", name="Resp Rate", line=dict(color="#6D597A", width=1.8)),
            row=5,
            col=1,
        )

        # Formatting: Vertical Line at t=0 and Shaded Regions
        for r in range(1, 6):
            # Stand transition line
            fig_wf.add_vline(x=0.0, line_dash="dash", line_color="#333333", line_width=2.0, row=r, col=1)
            # Shaded pre-stand and post-stand regions
            fig_wf.add_vrect(
                x0=-60.0,
                x1=0.0,
                fillcolor="#457B9D",
                opacity=0.06,
                layer="below",
                line_width=0,
                row=r,
                col=1,
            )
            fig_wf.add_vrect(
                x0=0.0,
                x1=60.0,
                fillcolor="#E63946",
                opacity=0.06,
                layer="below",
                line_width=0,
                row=r,
                col=1,
            )

        # Add Phase Labels on top panel
        fig_wf.add_annotation(
            x=-30.0,
            y=1.05,
            yref="paper",
            text="<b>Pre-Stand Seated Baseline (-60s to 0s)</b>",
            showarrow=False,
            font=dict(size=12, color="#457B9D"),
        )
        fig_wf.add_annotation(
            x=30.0,
            y=1.05,
            yref="paper",
            text="<b>Post-Stand Orthostatic Challenge (0s to +60s)</b>",
            showarrow=False,
            font=dict(size=12, color="#E63946"),
        )

        fig_wf.update_layout(
            height=850,
            template="plotly_white",
            margin=dict(l=60, r=40, t=60, b=40),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        fig_wf.update_xaxes(title_text="Time Relative to Stand Transition (seconds)", row=5, col=1)
        st.plotly_chart(fig_wf, use_container_width=True)


# =============================================================================
# TAB 4: 🧬 DIGITAL TWIN PREDICTION
# =============================================================================
elif selected_tab == "🧬 Digital Twin Prediction":
    st.title("🧬 Digital Twin Forecasting & Explainability")
    st.markdown("Forecast post-stand orthostatic MCA blood-flow velocity trajectory and quantify patient-specific vulnerability.")

    p1, p2 = st.columns([1, 2])
    with p1:
        selected_sid = st.selectbox(
            "Select Patient for In-Silico Calibration",
            options=subject_list,
            format_func=lambda s: subject_display_names.get(s, s),
            index=0,
            key="dt_pred_subj",
        )
        subj_meta = subjects_df[subjects_df["subject_id"] == selected_sid].iloc[0]
        subj_features = features_df.loc[[selected_sid]]
        subj_actual_targets = targets_df.loc[[selected_sid]] if selected_sid in targets_df.index else None

        # Display Tiered Digital Twin State
        st.markdown("#### Patient Digital Twin State")
        with st.expander("Tier 1: Static Demographics", expanded=True):
            st.write(
                {
                    "Age": int(subj_meta["age"]),
                    "Sex": subj_meta["sex"],
                    "Cohort": str(subj_meta["group"]).upper(),
                    "HbA1c": f"{float(subj_meta['hba1c']):.1f}%",
                    "BMI": f"{float(subj_meta['bmi']):.1f} kg/m²",
                    "Hypertension": "Yes" if subj_meta.get("hypertension", False) else "No",
                }
            )

        with st.expander("Tier 2: Resting Hemodynamics & HRV", expanded=False):
            st.write(
                {
                    "Resting SBP": f"{float(subj_meta.get('resting_sbp', 120)):.0f} mmHg",
                    "Resting DBP": f"{float(subj_meta.get('resting_dbp', 80)):.0f} mmHg",
                    "Resting MAP": f"{float(subj_meta.get('resting_map', 93)):.0f} mmHg",
                    "Resting HR": f"{float(subj_meta.get('resting_hr', 70)):.0f} bpm",
                    "Baseline RMSSD": f"{float(subj_meta.get('baseline_rmssd', 25)):.1f} ms",
                    "Baseline SDNN": f"{float(subj_meta.get('baseline_sdnn', 85)):.1f} ms",
                }
            )

        with st.expander("Tier 3: Continuous Dynamic Features", expanded=False):
            st.write(
                {
                    "Early MAP Slope": f"{float(subj_features['early_map_slope'].iloc[0]):.2f} mmHg/s",
                    "Early HR Slope": f"{float(subj_features['early_hr_slope'].iloc[0]):.2f} bpm/s",
                    "Early SBP Drop (5s)": f"{float(subj_features['early_sbp_drop_5s'].iloc[0]):.1f} mmHg",
                    "Baroreflex Proxy": f"{float(subj_features['baroreflex_sensitivity_proxy'].iloc[0]):.2f} bpm/mmHg",
                }
            )

    with p2:
        # Run XGBoost Twin Inference
        df_lower, df_median, df_upper = xgb_model.predict_with_uncertainty(subj_features)
        pred_drop = float(df_median["pct_mca_drop"].iloc[0])
        pred_drop_low = float(df_lower["pct_mca_drop"].iloc[0])
        pred_drop_up = float(df_upper["pct_mca_drop"].iloc[0])

        pred_nadir = float(df_median["mca_nadir_cms"].iloc[0])
        pred_rec_time = float(df_median["recovery_time_s"].iloc[0])
        pred_time_nadir = float(df_median["time_to_nadir_s"].iloc[0])

        act_drop = float(subj_actual_targets["pct_mca_drop"].iloc[0]) if subj_actual_targets is not None else np.nan
        act_nadir = float(subj_actual_targets["mca_nadir_cms"].iloc[0]) if subj_actual_targets is not None else np.nan
        act_rec_time = float(subj_actual_targets["recovery_time_s"].iloc[0]) if subj_actual_targets is not None else np.nan

        st.subheader("Forecasted Hemodynamic Biomarkers")
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        with c_kpi1:
            delta_str = f"Actual: {act_drop:.1f}%" if not np.isnan(act_drop) else None
            st.metric("MCAv Drop", f"{pred_drop:.1f}%", delta=delta_str, delta_color="inverse")
            st.caption(f"90% PI: [{pred_drop_low:.1f}%, {pred_drop_up:.1f}%]")
        with c_kpi2:
            delta_str = f"Actual: {act_nadir:.1f}" if not np.isnan(act_nadir) else None
            st.metric("MCAv Nadir", f"{pred_nadir:.1f} cm/s", delta=delta_str)
            st.caption(f"90% PI: [{float(df_lower['mca_nadir_cms'].iloc[0]):.1f}, {float(df_upper['mca_nadir_cms'].iloc[0]):.1f}]")
        with c_kpi3:
            st.metric("Time to Nadir", f"{pred_time_nadir:.1f} s")
            st.caption("Post-stand latency")
        with c_kpi4:
            delta_str = f"Actual: {act_rec_time:.1f}s" if not np.isnan(act_rec_time) else None
            st.metric("Recovery Time", f"{pred_rec_time:.1f} s", delta=delta_str, delta_color="inverse")
            st.caption("Time to >=90% baseline")

        # Determine Research Response Status
        is_high_deviation = (pred_drop > control_drop_90th) or (pred_drop > z_score_threshold_val)
        if is_high_deviation:
            z_score = (pred_drop - control_drop_mean) / max(control_drop_std, 1e-3)
            st.markdown(
                f"""
                <div class="status-box-deviation">
                    <h4 style="color: #c62828; margin: 0;">🚨 Research Response Status: HIGH DEVIATION</h4>
                    <p style="margin: 6px 0 0 0; color: #333;">
                        Predicted MCAv drop of <b>{pred_drop:.1f}%</b> exceeds the 90th percentile threshold of healthy controls 
                        (<b>{control_drop_90th:.1f}%</b>; z-score: <b>+{z_score:.2f}</b>). 
                        Reflects marked dynamic cerebral autoregulatory delay and vulnerability to orthostatic cerebral hypoperfusion.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-box-normal">
                    <h4 style="color: #2e7d32; margin: 0;">✅ Research Response Status: NORMAL VARIATION</h4>
                    <p style="margin: 6px 0 0 0; color: #333;">
                        Predicted MCAv drop of <b>{pred_drop:.1f}%</b> resides within the expected physiological reference 
                        envelope of healthy controls (threshold: <b>{control_drop_90th:.1f}%</b>). Arteriolar autoregulatory 
                        reserve is adequately counteracting gravitational venous pooling.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="disclaimer-box">⚠️ <b>MANDATORY RESEARCH DISCLAIMER:</b> FOR RESEARCH PURPOSES ONLY. NOT A CLINICAL DIAGNOSTIC TOOL. Predictions are computational in-silico estimates.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Trajectory Forecast & SHAP Attribution Section
    traj_left, shap_right = st.columns([3, 2])

    with traj_left:
        st.subheader("Predicted Post-Stand MCAv Trajectory (0 to 60s)")
        knot_times = np.array(config.get("model", {}).get("trajectory_times", list(range(0, 65, 5))))
        low_arr, med_arr, up_arr = xgb_model.predict_trajectory(subj_features)

        fig_traj = go.Figure()

        # 90% Confidence Interval Band (Lower -> Upper)
        fig_traj.add_trace(
            go.Scatter(
                x=np.concatenate([knot_times, knot_times[::-1]]),
                y=np.concatenate([up_arr[0], low_arr[0][::-1]]),
                fill="toself",
                fillcolor="rgba(69, 123, 157, 0.22)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="90% Prediction Interval",
            )
        )

        # Median Trajectory Line
        fig_traj.add_trace(
            go.Scatter(
                x=knot_times,
                y=med_arr[0],
                mode="lines+markers",
                name="Predicted Median MCAv",
                line=dict(color="#1D3557", width=3.0),
                marker=dict(size=6),
            )
        )

        # Overlay Ground Truth Waveform Trajectory if Available
        wf_subj = processed_wf.get(selected_sid)
        if wf_subj is not None and "mcav_cms" in wf_subj.columns:
            post_wf = wf_subj[wf_subj["time_s"] >= 0.0]
            # Resample actual MCAv at knot times
            interp_actual = np.interp(knot_times, post_wf["time_s"].values, post_wf["mcav_cms"].values)
            # Normalize to percentage of baseline to match or plot raw
            pre_wf = wf_subj[(wf_subj["time_s"] >= -60.0) & (wf_subj["time_s"] <= 0.0)]
            base_vel = float(pre_wf["mcav_cms"].mean()) if len(pre_wf) > 0 else 60.0
            interp_pct = (interp_actual / base_vel) * 100.0

            fig_traj.add_trace(
                go.Scatter(
                    x=knot_times,
                    y=interp_pct,
                    mode="lines+markers",
                    name="Actual Ground Truth MCAv",
                    line=dict(color="#E63946", width=2.5, dash="dash"),
                    marker=dict(size=7, symbol="diamond"),
                )
            )

        fig_traj.update_layout(
            template="plotly_white",
            height=380,
            xaxis_title="Time Post-Stand (seconds)",
            yaxis_title="MCA Blood-Flow Velocity (% of Seated Baseline)",
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with shap_right:
        st.subheader("Patient-Specific Physiological Attribution (SHAP)")
        st.caption("Contribution of peripheral features toward predicted MCAv Drop (%)")

        try:
            shap_values_matrix, expected_val = xgb_model.get_shap_values(subj_features)
            shap_row = shap_values_matrix[0]
            feat_names = xgb_model.feature_names_

            # Build explainability DataFrame
            df_shap = pd.DataFrame(
                {
                    "feature": feat_names,
                    "value": [float(subj_features[f].iloc[0]) for f in feat_names],
                    "shap_importance": shap_row,
                    "abs_shap": np.abs(shap_row),
                }
            ).sort_values(by="abs_shap", ascending=True).tail(10)

            # Manual Plotly Horizontal Bar Chart
            bar_colors = [COLOR_T2D if v > 0 else COLOR_CONTROL for v in df_shap["shap_importance"]]
            fig_shap = go.Figure(
                go.Bar(
                    x=df_shap["shap_importance"],
                    y=df_shap["feature"],
                    orientation="h",
                    marker=dict(color=bar_colors),
                    text=[f"{v:+.2f}%" for v in df_shap["shap_importance"]],
                    textposition="auto",
                )
            )
            fig_shap.update_layout(
                template="plotly_white",
                height=380,
                xaxis_title="SHAP Value (Impact on % MCAv Drop)",
                yaxis_title="",
                margin=dict(l=10, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("🔴 Red: Increases predicted drop (aggravating factor) | 🔵 Blue: Limits drop (protective factor)")
        except Exception as e:
            st.warning(f"SHAP attribution temporarily unavailable: {e}")


# =============================================================================
# TAB 5: 📈 MODEL EVALUATION
# =============================================================================
elif selected_tab == "📈 Model Evaluation":
    st.title("📈 Rigorous Subject-Level Model Evaluation")
    st.markdown(
        """
        ### Zero-Leakage GroupKFold Benchmark (5 Folds Grouped by Subject)
        Validation guarantees strict subject-level isolation: no telemetry or static records from any subject 
        ever appear simultaneously in both training and test sets.
        """
    )

    with st.spinner("Computing Subject-Level Cross-Validation Benchmark..."):
        eval_dict = run_cached_evaluation(features_df, targets_df, config)

    comp_df = eval_dict["comparison_df"]
    xgb_preds = eval_dict["xgb_results"]["predictions"]
    y_true = eval_dict["xgb_results"]["y_true"]

    st.subheader("Model Comparison Benchmark: Model 0 vs Model 1 vs Model 2")
    st.dataframe(comp_df, use_container_width=True)

    # Bar chart comparing MAE across models
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.subheader("Error Comparison: % MCAv Drop (MAE)")
        mae_data = []
        for res_key, name in [
            ("mean_results", "Mean Baseline (M0)"),
            ("ridge_results", "Ridge Linear (M1)"),
            ("xgb_results", "XGBoost Twin (M2)"),
        ]:
            val = eval_dict[res_key]["metrics"].get("pct_mca_drop", {}).get("mae", 0.0)
            mae_data.append({"Model": name, "MAE (%)": val})
        df_mae = pd.DataFrame(mae_data)

        fig_mae = px.bar(
            df_mae,
            x="Model",
            y="MAE (%)",
            color="Model",
            color_discrete_sequence=[COLOR_CONTROL, COLOR_WARNING, COLOR_T2D],
            title="Mean Absolute Error on MCAv Drop (%) - Lower is Better",
        )
        fig_mae.update_layout(template="plotly_white", height=320, showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    with c_m2:
        st.subheader("Trajectory Knot Fit: Overall Trajectory (RMSE)")
        traj_data = []
        for res_key, name in [
            ("mean_results", "Mean Baseline (M0)"),
            ("ridge_results", "Ridge Linear (M1)"),
            ("xgb_results", "XGBoost Twin (M2)"),
        ]:
            val = eval_dict[res_key]["metrics"].get("overall_trajectory", {}).get("rmse", 0.0)
            traj_data.append({"Model": name, "RMSE (% velocity)": val})
        df_traj = pd.DataFrame(traj_data)

        fig_traj_bar = px.bar(
            df_traj,
            x="Model",
            y="RMSE (% velocity)",
            color="Model",
            color_discrete_sequence=[COLOR_CONTROL, COLOR_WARNING, COLOR_T2D],
            title="Continuous Trajectory Knot RMSE (%) - Lower is Better",
        )
        fig_traj_bar.update_layout(template="plotly_white", height=320, showlegend=False)
        st.plotly_chart(fig_traj_bar, use_container_width=True)

    st.markdown("---")

    # Scatter: Predicted vs Actual for pct_mca_drop
    s_left, s_right = st.columns(2)
    with s_left:
        st.subheader("Out-of-Fold Calibration: % MCAv Drop")
        scatter_df = pd.DataFrame(
            {
                "subject_id": y_true.index,
                "Actual_Drop": y_true["pct_mca_drop"].values,
                "Predicted_Drop": xgb_preds["pct_mca_drop"].values,
                "Group": [
                    str(subjects_df.loc[subjects_df["subject_id"] == sid, "group"].iloc[0]).upper()
                    if sid in subjects_df["subject_id"].values
                    else "T2D"
                    for sid in y_true.index
                ],
            }
        )

        min_val = min(scatter_df["Actual_Drop"].min(), scatter_df["Predicted_Drop"].min()) - 2.0
        max_val = max(scatter_df["Actual_Drop"].max(), scatter_df["Predicted_Drop"].max()) + 2.0

        fig_scatter = px.scatter(
            scatter_df,
            x="Actual_Drop",
            y="Predicted_Drop",
            color="Group",
            color_discrete_map={"T2D": COLOR_T2D, "CONTROL": COLOR_CONTROL},
            hover_name="subject_id",
            labels={"Actual_Drop": "Actual MCAv Drop (%)", "Predicted_Drop": "Digital Twin Predicted Drop (%)"},
            title="Predicted vs Actual MCAv Drop (% Baseline)",
        )
        # Identity Line
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="1:1 Identity Line",
                line=dict(color="#6c757d", dash="dash"),
            )
        )
        fig_scatter.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with s_right:
        st.subheader("Residual Diagnostic Distribution")
        scatter_df["Residual"] = scatter_df["Actual_Drop"] - scatter_df["Predicted_Drop"]
        fig_resid = px.histogram(
            scatter_df,
            x="Residual",
            nbins=14,
            color="Group",
            color_discrete_map={"T2D": COLOR_T2D, "CONTROL": COLOR_CONTROL},
            labels={"Residual": "Residual Error (Actual - Predicted, %)"},
            title="Residual Error Distribution (Zero-Centered Validation)",
        )
        fig_resid.add_vline(x=0.0, line_dash="solid", line_color="#000", line_width=1.5)
        fig_resid.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig_resid, use_container_width=True)


# =============================================================================
# TAB 6: 🔮 WHAT-IF SIMULATOR
# =============================================================================
elif selected_tab == "🔮 What-If Simulator":
    st.title("🔮 In-Silico Physiological What-If Simulator")
    st.markdown("Interrogate the digital twin by perturbing clinical and hemodynamic parameters to evaluate counterfactual interventions.")

    simulator = WhatIfSimulator(xgb_model, get_feature_names())

    selected_sid = st.selectbox(
        "Select Base Patient to Simulate",
        options=subject_list,
        format_func=lambda s: subject_display_names.get(s, s),
        index=0,
        key="sim_subj_select",
    )

    base_series = features_df.loc[selected_sid]
    orig_meta = subjects_df[subjects_df["subject_id"] == selected_sid].iloc[0]

    st.subheader("1. Physiological Preset Scenarios")
    presets = simulator.get_preset_scenarios()
    preset_cols = st.columns(len(presets))

    # Initialize session state for interactive adjustments
    if "sim_hba1c" not in st.session_state:
        st.session_state["sim_hba1c"] = float(base_series.get("hba1c", 7.5))
    if "sim_sbp" not in st.session_state:
        st.session_state["sim_sbp"] = float(base_series.get("resting_sbp", 130.0))
    if "sim_dbp" not in st.session_state:
        st.session_state["sim_dbp"] = float(base_series.get("resting_dbp", 80.0))
    if "sim_rmssd" not in st.session_state:
        st.session_state["sim_rmssd"] = float(base_series.get("baseline_rmssd", 25.0))
    if "sim_age" not in st.session_state:
        st.session_state["sim_age"] = int(base_series.get("age", 65))

    for idx, preset in enumerate(presets):
        with preset_cols[idx]:
            if st.button(preset["name"], key=f"btn_preset_{idx}"):
                mods = preset["modifications"]
                if "hba1c" in mods:
                    st.session_state["sim_hba1c"] = float(mods["hba1c"])
                if "resting_sbp" in mods:
                    st.session_state["sim_sbp"] = float(mods["resting_sbp"])
                if "resting_dbp" in mods:
                    st.session_state["sim_dbp"] = float(mods["resting_dbp"])
                if "baseline_rmssd" in mods:
                    st.session_state["sim_rmssd"] = float(mods["baseline_rmssd"])
                if "younger" in preset["name"].lower():
                    st.session_state["sim_age"] = max(18, int(base_series.get("age", 65)) - 10)

    st.markdown("---")
    st.subheader("2. Interactive Counterfactual Parameter Tuning")

    col_tune1, col_tune2, col_tune3, col_tune4 = st.columns(4)
    with col_tune1:
        sim_hba1c = st.slider("Glycated Hemoglobin (HbA1c %)", 4.5, 12.0, value=float(st.session_state["sim_hba1c"]), step=0.1)
    with col_tune2:
        sim_sbp = st.slider("Resting Systolic BP (mmHg)", 90.0, 190.0, value=float(st.session_state["sim_sbp"]), step=1.0)
    with col_tune3:
        sim_dbp = st.slider("Resting Diastolic BP (mmHg)", 50.0, 115.0, value=float(st.session_state["sim_dbp"]), step=1.0)
    with col_tune4:
        sim_rmssd = st.slider("Autonomic Vagal Tone (RMSSD ms)", 5.0, 100.0, value=float(st.session_state["sim_rmssd"]), step=1.0)

    # Compute MAP
    sim_map = sim_dbp + (sim_sbp - sim_dbp) / 3.0

    modifications = {
        "hba1c": sim_hba1c,
        "resting_sbp": sim_sbp,
        "resting_dbp": sim_dbp,
        "resting_map": sim_map,
        "baseline_rmssd": sim_rmssd,
        "age": st.session_state["sim_age"],
    }

    sim_result = simulator.simulate(base_series, modifications)
    orig_p = sim_result["original_prediction"]
    mod_p = sim_result["modified_prediction"]
    delta_p = sim_result["delta"]

    st.markdown("---")
    st.subheader("3. Simulation Impact on Orthostatic MCAv Stability")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        d_val = delta_p.get("pct_mca_drop", 0.0)
        st.metric(
            "Forecasted MCAv Drop",
            f"{mod_p.get('pct_mca_drop', 0.0):.1f}%",
            delta=f"{d_val:+.1f}% vs Baseline",
            delta_color="inverse",
        )
    with r2:
        d_nadir = delta_p.get("mca_nadir_cms", 0.0)
        st.metric(
            "Forecasted MCAv Nadir",
            f"{mod_p.get('mca_nadir_cms', 0.0):.1f} cm/s",
            delta=f"{d_nadir:+.1f} cm/s",
        )
    with r3:
        d_rec = delta_p.get("recovery_time_s", 0.0)
        st.metric(
            "Forecasted Recovery Time",
            f"{mod_p.get('recovery_time_s', 0.0):.1f} s",
            delta=f"{d_rec:+.1f} s",
            delta_color="inverse",
        )
    with r4:
        is_mod_high = mod_p.get("pct_mca_drop", 0.0) > control_drop_90th
        st.markdown(
            f"""
            <div style="background-color: {'#ffebee' if is_mod_high else '#e8f5e9'}; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="font-size: 0.8rem; color: #555;">SIMULATED STATUS</div>
                <div style="font-weight: 700; color: {'#c62828' if is_mod_high else '#2e7d32'};">
                    {'HIGH DEVIATION' if is_mod_high else 'NORMAL VARIATION'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sensitivity Curve Section
    st.markdown("---")
    st.subheader("4. Parameter Sensitivity Response Curve")
    sens_feat = st.selectbox(
        "Select Parameter to Sweep for Response Elasticity",
        ["hba1c", "resting_sbp", "baseline_rmssd", "age", "bmi"],
        index=0,
    )

    feat_ranges = {
        "hba1c": np.linspace(5.0, 11.5, 25),
        "resting_sbp": np.linspace(100.0, 180.0, 25),
        "baseline_rmssd": np.linspace(10.0, 80.0, 25),
        "age": np.linspace(50.0, 85.0, 25),
        "bmi": np.linspace(20.0, 42.0, 25),
    }

    sweep_vals = feat_ranges.get(sens_feat, np.linspace(10, 100, 20))
    df_sens = simulator.sensitivity_analysis(base_series, sens_feat, sweep_vals)

    fig_sens = px.line(
        df_sens,
        x=sens_feat,
        y="pct_mca_drop",
        labels={"pct_mca_drop": "Forecasted MCAv Drop (%)", sens_feat: f"Perturbed {sens_feat}"},
        title=f"Predicted MCAv Drop (%) vs {sens_feat} Elasticity",
    )
    fig_sens.add_hline(
        y=control_drop_90th,
        line_dash="dash",
        line_color="#E63946",
        annotation_text="90th Percentile Control Threshold",
    )
    # Add vertical line for patient's actual baseline value
    curr_patient_val = float(base_series.get(sens_feat, sweep_vals[0]))
    fig_sens.add_vline(
        x=curr_patient_val,
        line_dash="dot",
        line_color="#1D3557",
        annotation_text="Patient Baseline",
    )
    fig_sens.update_layout(template="plotly_white", height=340)
    st.plotly_chart(fig_sens, use_container_width=True)


# =============================================================================
# TAB 7: 📋 RESEARCH REPORT
# =============================================================================
elif selected_tab == "📋 Research Report":
    st.title("📋 CerebroTwin In-Silico Research Report")
    st.markdown("Comprehensive individualized physiological evaluation for biomedical researchers.")

    selected_sid = st.selectbox(
        "Select Subject to Generate Report",
        options=subject_list,
        format_func=lambda s: subject_display_names.get(s, s),
        index=0,
        key="rep_subj_select",
    )

    subj_meta = subjects_df[subjects_df["subject_id"] == selected_sid].iloc[0]
    subj_features = features_df.loc[[selected_sid]]
    subj_targets = targets_df.loc[[selected_sid]] if selected_sid in targets_df.index else None

    # Predict with Uncertainty
    df_low, df_med, df_up = xgb_model.predict_with_uncertainty(subj_features)
    pred_drop = float(df_med["pct_mca_drop"].iloc[0])
    pred_nadir = float(df_med["mca_nadir_cms"].iloc[0])
    pred_rec = float(df_med["recovery_time_s"].iloc[0])
    pred_latency = float(df_med["time_to_nadir_s"].iloc[0])

    is_high_dev = (pred_drop > control_drop_90th) or (pred_drop > z_score_threshold_val)
    status_label = "HIGH DEVIATION" if is_high_dev else "NORMAL VARIATION"

    # Assemble Structured Report Text
    report_text = f"""================================================================================
CEREBROTWIN DIGITAL TWIN PHYSIOLOGICAL RESEARCH REPORT
Personalized In-Silico Evaluation of Orthostatic MCAv Instability
================================================================================

SUBJECT IDENTIFIER: {selected_sid}
RESEARCH COHORT:    {str(subj_meta['group']).upper()}
BIOLOGICAL SEX:     {subj_meta['sex']}
AGE:                {int(subj_meta['age'])} years
BODY MASS INDEX:    {float(subj_meta['bmi']):.1f} kg/m²
GLYCATED HBA1C:     {float(subj_meta['hba1c']):.1f}%
HYPERTENSION:       {'Yes' if subj_meta.get('hypertension', False) else 'No'}

--------------------------------------------------------------------------------
1. RESTING HEMODYNAMIC & AUTONOMIC PROFILE (SEATED BASELINE, t = -60s to 0s)
--------------------------------------------------------------------------------
Resting Systolic Blood Pressure:   {float(subj_meta.get('resting_sbp', 120)):.0f} mmHg
Resting Diastolic Blood Pressure:  {float(subj_meta.get('resting_dbp', 80)):.0f} mmHg
Resting Mean Arterial Pressure:    {float(subj_meta.get('resting_map', 93)):.0f} mmHg
Resting Heart Rate:                {float(subj_meta.get('resting_hr', 70)):.0f} bpm
Autonomic Tone (RMSSD):            {float(subj_meta.get('baseline_rmssd', 25)):.1f} ms
Total HRV Power (SDNN):            {float(subj_meta.get('baseline_sdnn', 85)):.1f} ms

--------------------------------------------------------------------------------
2. DIGITAL TWIN POST-STAND FORECAST (ACTIVE STAND, t = 0s to +60s)
--------------------------------------------------------------------------------
Predicted % MCAv Drop:             {pred_drop:.1f}% [90% PI: {float(df_low['pct_mca_drop'].iloc[0]):.1f}% - {float(df_up['pct_mca_drop'].iloc[0]):.1f}%]
Predicted MCAv Nadir:              {pred_nadir:.1f} cm/s [90% PI: {float(df_low['mca_nadir_cms'].iloc[0]):.1f} - {float(df_up['mca_nadir_cms'].iloc[0]):.1f} cm/s]
Predicted Latency to Nadir:        {pred_latency:.1f} seconds
Predicted Autoregulatory Recovery: {pred_rec:.1f} seconds

--------------------------------------------------------------------------------
3. BIOMARKER ANOMALY EVALUATION
--------------------------------------------------------------------------------
Research Response Status:          {status_label}
Control Cohort 90th Percentile:    {control_drop_90th:.1f}% MCAv Drop
Control Mean ± 1.65 SD (Z=1.65):   {z_score_threshold_val:.1f}% MCAv Drop

INTERPRETATION:
The participant's predicted middle cerebral artery blood-flow velocity (MCAv)
drop of {pred_drop:.1f}% {'exceeds' if is_high_dev else 'remains within'} the healthy reference envelope.
{'This indicates elevated risk of acute cerebral hypoperfusion triggered by blunted myogenic autoregulation.' if is_high_dev else 'Dynamic cerebral autoregulatory dilation is operating within expected physiological bounds.'}

================================================================================
DISCLAIMER:
FOR RESEARCH PURPOSES ONLY. NOT A CLINICAL DIAGNOSTIC TOOL.
CerebroTwin outputs represent mathematical in-silico simulations generated from
peripheral hemodynamic telemetry and machine learning models.
================================================================================
"""

    st.text_area("Generated Scientific Report", value=report_text, height=450)

    # Download Button
    st.download_button(
        label="📥 Download Research Report (.txt)",
        data=report_text,
        file_name=f"CerebroTwin_Report_{selected_sid}.txt",
        mime="text/plain",
    )
