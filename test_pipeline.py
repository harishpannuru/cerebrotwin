"""End-to-end pipeline validation script for CerebroTwin."""
import sys
sys.path.insert(0, ".")
import yaml
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("CerebroTwin End-to-End Pipeline Validation")
print("=" * 60)

# 1. Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)
print("\n[1] Config loaded OK")

# 2. Load data
from src.data.loader import CerebroTwinDataLoader
loader = CerebroTwinDataLoader(config)
subjects_df, waveforms = loader.load()
print(f"[2] Data loaded: {len(subjects_df)} subjects, {len(waveforms)} waveform sets")

# 3. Preprocess
from src.data.preprocessing import WaveformPreprocessor
preprocessor = WaveformPreprocessor(config)
waveforms = preprocessor.process(waveforms)
print(f"[3] Preprocessing OK: {len(waveforms)} subjects processed")

# 4. Extract features and targets
from src.features.feature_engineering import (
    extract_features_for_cohort, extract_targets_for_cohort,
    get_feature_names, get_target_names, get_trajectory_target_names
)
X = extract_features_for_cohort(subjects_df, waveforms, config)
y = extract_targets_for_cohort(subjects_df, waveforms, config)
print(f"[4] Features: {X.shape}, Targets: {y.shape}")
print(f"    Feature names: {get_feature_names()[:5]}...")
print(f"    Target names: {get_target_names()}")
print(f"    Trajectory knots: {len(get_trajectory_target_names())}")

# 5. Train models
from src.models.baselines import MeanBaselineModel, RidgeBaselineModel
from src.models.main_xgboost import XGBoostTwinModel

m0 = MeanBaselineModel()
m0.fit(X, y)
p0 = m0.predict(X)
print(f"\n[5a] {m0.get_name()}: predict shape={p0.shape} OK")

m1 = RidgeBaselineModel(alpha=10.0)
m1.fit(X, y)
p1 = m1.predict(X)
print(f"[5b] {m1.get_name()}: predict shape={p1.shape} OK")

m2 = XGBoostTwinModel(config)
m2.fit(X, y)
p2 = m2.predict(X)
print(f"[5c] {m2.get_name()}: predict shape={p2.shape} OK")

# 6. Uncertainty estimation
lo, med, hi = m2.predict_with_uncertainty(X.iloc[:3])
print(f"\n[6] Uncertainty bands: lower={lo.shape}, median={med.shape}, upper={hi.shape}")

# 7. Trajectory prediction
traj_lo, traj_med, traj_hi = m2.predict_trajectory(X.iloc[:1])
print(f"[7] Trajectory: lower={traj_lo.shape}, median={traj_med.shape}, upper={traj_hi.shape}")

# 8. SHAP explainability
shap_vals, exp_val = m2.get_shap_values(X.iloc[:3])
print(f"[8] SHAP values: shape={shap_vals.shape}, expected_value={exp_val:.2f}")

# 9. Feature importance
fi = m2.get_feature_importance()
top_feat = fi.iloc[0]["feature"] if "feature" in fi.columns else fi.index[0]
print(f"[9] Feature importance: {len(fi)} features, top feature: {top_feat}")

# 10. Model evaluation (GroupKFold)
from src.models.evaluator import ModelEvaluator
evaluator = ModelEvaluator(config)
groups = X.index.to_series()
results_m2 = evaluator.evaluate_group_kfold(m2, X, y, groups, n_splits=5)
report = evaluator.generate_report(results_m2)
print(f"\n[10] GroupKFold evaluation:")
print(report.to_string())

# 11. Counterfactual simulation
from src.models.counterfactuals import WhatIfSimulator
simulator = WhatIfSimulator(m2, list(X.columns))
base = X.iloc[0]
result = simulator.simulate(base, {"hba1c": 7.0})
orig_drop = result["original_prediction"].get("pct_mca_drop", "N/A")
mod_drop = result["modified_prediction"].get("pct_mca_drop", "N/A")
print(f"\n[11] What-If: HbA1c -> 7.0")
print(f"     Original pct_mca_drop: {orig_drop}")
print(f"     Modified pct_mca_drop: {mod_drop}")

# 12. Digital Twin State
from src.features.digital_twin_state import DigitalTwinState
meta = subjects_df.iloc[0].to_dict()
sid = meta.get("subject_id", X.index[0])
wf = waveforms.get(sid, list(waveforms.values())[0])
twin = DigitalTwinState(meta, wf)
features_dict = twin.get_all_features(wf, t_current=0.0)
print(f"\n[12] DigitalTwinState for {sid}: {len(features_dict)} features")

print("\n" + "=" * 60)
print("ALL 12 PIPELINE STAGES PASSED SUCCESSFULLY")
print("=" * 60)
