"""
CerebroTwin - Personalized Digital Twin Pipeline CLI
Forecasting Orthostatic Cerebral Blood-Flow-Velocity Instability in Type 2 Diabetes

CRITICAL DOMAIN RULES:
1. ALWAYS use "MCA blood-flow velocity" or "MCAv" (cm/s), NEVER "cerebral blood flow" or volumetric perfusion.
2. MCAv (TCD) is GROUND TRUTH for training ONLY — NEVER a runtime input feature.
3. Abnormality flags: "Research Response Status: HIGH DEVIATION" — NEVER clinical diagnosis.
4. t=0 is the sit-to-stand transition moment. Continuous waveforms span -60s to +60s.
5. Subject-level splitting only (GroupKFold / LeaveOneGroupOut by subject_id).
6. N=39 subjects: 20 T2D, 19 Controls, aged 50-85.

Usage:
  python main.py generate    # Generate synthetic cohort data
  python main.py train       # Train all baseline and production models
  python main.py evaluate    # Run subject-level cross-validation benchmark
  python main.py dashboard   # Launch interactive Streamlit dashboard
  python main.py run         # Full end-to-end pipeline
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
import yaml

# Set up project root in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import CerebroTwinDataLoader
from src.data.preprocessing import WaveformPreprocessor
from src.features.feature_engineering import (
    extract_features_for_cohort,
    extract_targets_for_cohort,
    get_feature_names,
    get_target_names,
    get_trajectory_target_names,
)
from src.models.baselines import MeanBaselineModel, RidgeBaselineModel
from src.models.evaluator import ModelEvaluator
from src.models.main_xgboost import XGBoostTwinModel


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_generate(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Generate or verify cohort data and multimodal waveforms."""
    print("\n" + "=" * 70)
    print("STEP 1: INGESTING / GENERATING PHYSIOLOGICAL COHORT")
    print("=" * 70)
    print("Protocol: 120s Sit-to-Stand (-60s to +60s) @ 50Hz")
    print("Biomarker: Middle Cerebral Artery blood-flow velocity (MCAv in cm/s, Ground Truth)")

    loader = CerebroTwinDataLoader(config)
    subjects_df, waveforms = loader.load()

    n_t2d = len(subjects_df[subjects_df["group"].astype(str).str.upper() == "T2D"])
    n_ctl = len(subjects_df[subjects_df["group"].astype(str).str.upper() == "CONTROL"])
    print(f"[OK] Ingested {len(subjects_df)} subjects: {n_t2d} T2D, {n_ctl} Controls.")
    print(f"[OK] Waveforms available for {len(waveforms)} subjects.")
    
    first_sid = next(iter(waveforms))
    first_wf = waveforms[first_sid]
    print(f"[OK] Sample waveform channels for {first_sid}: {list(first_wf.columns)}")
    print(f"[OK] Waveform duration: {first_wf['time_s'].min():.1f}s to {first_wf['time_s'].max():.1f}s")
    return subjects_df, waveforms


def cmd_train(
    config: Dict[str, Any],
    subjects_df: Optional[pd.DataFrame] = None,
    waveforms: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[MeanBaselineModel, RidgeBaselineModel, XGBoostTwinModel]:
    """Preprocess waveforms, engineer features, and train all 3 physiological models."""
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING, FEATURE EXTRACTION, AND MODEL TRAINING")
    print("=" * 70)

    if subjects_df is None or waveforms is None:
        loader = CerebroTwinDataLoader(config)
        subjects_df, waveforms = loader.load()

    # Preprocessing
    print("[1/4] Preprocessing continuous physiological waveforms (filtering & alignment)...")
    preprocessor = WaveformPreprocessor(config)
    processed_waveforms = preprocessor.process(waveforms)

    # Feature and Target Extraction
    print("[2/4] Extracting 27 non-invasive peripheral features (Zero MCAv leakage)...")
    features_df = extract_features_for_cohort(subjects_df, processed_waveforms, config)
    print(f"[OK] Extracted features matrix: shape {features_df.shape}")

    print("[3/4] Extracting Ground Truth MCA blood-flow velocity (MCAv) targets...")
    targets_df = extract_targets_for_cohort(subjects_df, processed_waveforms, config)
    print(f"[OK] Extracted targets matrix: shape {targets_df.shape}")

    # Model Training
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[4/4] Training Models:")
    print("  - Model 0: Mean Baseline Predictor...")
    mean_model = MeanBaselineModel()
    mean_model.fit(features_df, targets_df)
    joblib.dump(mean_model, models_dir / "mean_baseline.joblib")

    print("  - Model 1: Ridge Regularized Linear Predictor (alpha=10.0)...")
    ridge_alpha = float(config.get("model", {}).get("ridge_alpha", 10.0))
    ridge_model = RidgeBaselineModel(alpha=ridge_alpha)
    ridge_model.fit(features_df, targets_df)
    joblib.dump(ridge_model, models_dir / "ridge_baseline.joblib")

    print("  - Model 2: XGBoost Digital Twin (Production, Quantile Uncertainty)...")
    xgb_model = XGBoostTwinModel(config)
    xgb_model.fit(features_df, targets_df)
    joblib.dump(xgb_model, models_dir / "xgboost_twin.joblib")

    # Save feature and target column definitions
    joblib.dump(get_feature_names(), models_dir / "feature_names.joblib")
    joblib.dump(get_target_names() + get_trajectory_target_names(), models_dir / "target_names.joblib")

    print(f"[OK] All models and feature dictionaries persisted to {models_dir.resolve()}")
    return mean_model, ridge_model, xgb_model


def cmd_evaluate(
    config: Dict[str, Any],
    subjects_df: Optional[pd.DataFrame] = None,
    waveforms: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Run subject-level GroupKFold cross-validation across all models."""
    print("\n" + "=" * 70)
    print("STEP 3: SUBJECT-LEVEL CROSS-VALIDATION EVALUATION (GroupKFold)")
    print("=" * 70)
    print("Enforcing strict subject isolation: 5 folds grouped by subject_id.")

    if subjects_df is None or waveforms is None:
        loader = CerebroTwinDataLoader(config)
        subjects_df, waveforms = loader.load()

    preprocessor = WaveformPreprocessor(config)
    processed_waveforms = preprocessor.process(waveforms)
    features_df = extract_features_for_cohort(subjects_df, processed_waveforms, config)
    targets_df = extract_targets_for_cohort(subjects_df, processed_waveforms, config)

    evaluator = ModelEvaluator(config)
    groups = subjects_df["subject_id"].to_numpy()

    # Instantiate fresh models for CV
    mean_model = MeanBaselineModel()
    ridge_alpha = float(config.get("model", {}).get("ridge_alpha", 10.0))
    ridge_model = RidgeBaselineModel(alpha=ridge_alpha)
    xgb_model = XGBoostTwinModel(config)

    print("Evaluating Model 0 (Mean Baseline)...")
    mean_res = evaluator.evaluate_group_kfold(mean_model, features_df, targets_df, groups=groups, n_splits=5)

    print("Evaluating Model 1 (Ridge Linear)...")
    ridge_res = evaluator.evaluate_group_kfold(ridge_model, features_df, targets_df, groups=groups, n_splits=5)

    print("Evaluating Model 2 (XGBoost Digital Twin)...")
    xgb_res = evaluator.evaluate_group_kfold(xgb_model, features_df, targets_df, groups=groups, n_splits=5)

    comp_df = evaluator.compare_models([mean_res, ridge_res, xgb_res])

    print("\n" + "-" * 70)
    print("CROSS-VALIDATION BENCHMARK COMPARISON TABLE")
    print("-" * 70)
    print(comp_df.to_string(index=False))

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(models_dir / "evaluation_results.csv", index=False)
    eval_dict = {
        "mean_results": mean_res,
        "ridge_results": ridge_res,
        "xgb_results": xgb_res,
        "comparison_df": comp_df,
    }
    joblib.dump(eval_dict, models_dir / "eval_dict.joblib")
    print(f"\n[OK] Evaluation report saved to {models_dir / 'evaluation_results.csv'}")

    return comp_df


def cmd_dashboard() -> None:
    """Launch the interactive Streamlit dashboard."""
    dashboard_script = project_root / "src" / "app" / "dashboard.py"
    if not dashboard_script.exists():
        raise FileNotFoundError(f"Dashboard script not found at {dashboard_script}")

    print("\n" + "=" * 70)
    print("LAUNCHING CEREBROTWIN STREAMLIT SCIENTIFIC DASHBOARD")
    print("=" * 70)
    print(f"Target file: {dashboard_script.resolve()}")
    print("Opening browser session...")

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_script)])


def main() -> None:
    """CLI Parser and Dispatcher."""
    parser = argparse.ArgumentParser(
        description="CerebroTwin: Personalized Digital Twin for Forecasting Orthostatic MCAv Instability in T2D.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    subparsers.add_parser("generate", help="Generate or load physiological cohort data")
    subparsers.add_parser("train", help="Train baseline and production XGBoost models")
    subparsers.add_parser("evaluate", help="Run subject-level GroupKFold cross-validation")
    subparsers.add_parser("dashboard", help="Launch interactive Streamlit dashboard")
    subparsers.add_parser("run", help="Full end-to-end pipeline: generate + train + evaluate + dashboard")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    config = load_config()

    if args.subcommand == "generate":
        cmd_generate(config)
    elif args.subcommand == "train":
        cmd_train(config)
    elif args.subcommand == "evaluate":
        cmd_evaluate(config)
    elif args.subcommand == "dashboard":
        cmd_dashboard()
    elif args.subcommand == "run":
        subjects_df, waveforms = cmd_generate(config)
        cmd_train(config, subjects_df=subjects_df, waveforms=waveforms)
        cmd_evaluate(config, subjects_df=subjects_df, waveforms=waveforms)
        cmd_dashboard()


if __name__ == "__main__":
    main()
