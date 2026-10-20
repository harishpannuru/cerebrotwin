"""
CerebroTwin - Physiological What-If Simulation and Counterfactual Engine

Enables clinician-researchers to explore in-silico physiological counterfactuals:
  - "What if this patient's HbA1c were optimized to 7.0%?"
  - "What if blood pressure were strictly normalized to 120/75 mmHg?"
  - "What if autonomic vagal tone (RMSSD/SDNN) were enhanced?"
  - "What if this patient were 10 years younger?"

CRITICAL DOMAIN RULES:
- ALWAYS use "MCA blood-flow velocity" or "MCAv" (cm/s), NEVER "cerebral blood flow".
- MCAv is GROUND TRUTH ONLY — NEVER an input feature.
- Abnormality classification: "Research Response Status: HIGH DEVIATION" — NEVER clinical diagnosis.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class WhatIfSimulator:
    """
    Physiological Digital Twin What-If Simulation Engine.
    
    Perturbs peripheral physiological, metabolic, and autonomic state features
    and interrogates the trained digital twin to forecast resulting deviations
    in orthostatic MCA blood-flow velocity stability.
    """

    def __init__(self, model: Any, feature_names: List[str]) -> None:
        """
        Initialize WhatIfSimulator.
        
        Args:
            model: Trained forecaster (e.g., XGBoostTwinModel, RidgeBaselineModel).
            feature_names: List of all 27 expected feature names.
        """
        self.model = model
        self.feature_names = list(feature_names)

    def _predict_single(self, features_series: pd.Series) -> Dict[str, float]:
        """Generate prediction dictionary for a single sample Series."""
        row_dict = features_series.to_dict()
        # Ensure all required features are present
        aligned_dict = {f: float(row_dict.get(f, 0.0)) for f in self.feature_names}
        df_input = pd.DataFrame([aligned_dict], columns=self.feature_names)
        
        preds = self.model.predict(df_input)
        if isinstance(preds, pd.DataFrame):
            return {col: float(preds.iloc[0][col]) for col in preds.columns}
        elif isinstance(preds, np.ndarray):
            target_names = getattr(self.model, "target_names_", [f"target_{i}" for i in range(preds.shape[1])])
            return {target_names[i]: float(preds[0, i]) for i in range(len(target_names))}
        return {}

    def simulate(
        self, base_features: pd.Series, modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply physiological modifications to base patient features and predict outcomes.
        
        Args:
            base_features: Original patient feature vector (pd.Series).
            modifications: Dictionary mapping feature names to counterfactual target values.
            
        Returns:
            Dictionary with:
              - 'original_prediction': dict of baseline predicted targets
              - 'modified_prediction': dict of counterfactual predicted targets
              - 'delta': dict of changes (modified - original)
              - 'modified_features': pd.Series of the counterfactual patient state
        """
        mod_features = base_features.copy()

        # Apply specific modifications
        for feat, val in modifications.items():
            if feat == "age_offset":
                curr_age = float(mod_features.get("age", 65.0))
                mod_features["age"] = max(18.0, curr_age + float(val))
            else:
                mod_features[feat] = float(val)

        orig_pred = self._predict_single(base_features)
        mod_pred = self._predict_single(mod_features)

        delta = {}
        for target, val_mod in mod_pred.items():
            val_orig = orig_pred.get(target, 0.0)
            delta[target] = float(val_mod - val_orig)

        return {
            "original_prediction": orig_pred,
            "modified_prediction": mod_pred,
            "delta": delta,
            "modified_features": mod_features,
        }

    def compare_scenarios(
        self, base_features: pd.Series, scenarios: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple what-if physiological scenarios side-by-side against baseline.
        
        Args:
            base_features: Original patient feature vector (pd.Series).
            scenarios: List of scenario dictionaries, each containing:
                       - 'name': str (e.g., 'Improved Glycemic Control')
                       - 'modifications': dict of feature modifications
                       
        Returns:
            pd.DataFrame comparing baseline state and each scenario's forecasted outcomes.
        """
        orig_pred = self._predict_single(base_features)

        # Baseline reference row
        rows: List[Dict[str, Any]] = [
            {
                "Scenario": "Baseline (Current State)",
                "pct_mca_drop (%)": round(orig_pred.get("pct_mca_drop", 0.0), 2),
                "Δ Drop (%)": 0.0,
                "mca_nadir (cm/s)": round(orig_pred.get("mca_nadir_cms", 0.0), 2),
                "Δ Nadir (cm/s)": 0.0,
                "recovery_time (s)": round(orig_pred.get("recovery_time_s", 0.0), 2),
                "Δ Rec Time (s)": 0.0,
                "deficit_auc": round(orig_pred.get("cumulative_deficit_auc", 0.0), 2),
                "Δ Deficit AUC": 0.0,
            }
        ]

        for sc in scenarios:
            name = sc.get("name", "Scenario")
            mod_dict = dict(sc.get("modifications", {}))

            # Dynamically compute younger age scenario
            if "younger" in name.lower() or name.startswith("Younger"):
                curr_age = float(base_features.get("age", 65.0))
                mod_dict["age"] = max(18.0, curr_age - 10.0)

            sim_res = self.simulate(base_features, mod_dict)
            mp = sim_res["modified_prediction"]
            dl = sim_res["delta"]

            rows.append({
                "Scenario": name,
                "pct_mca_drop (%)": round(mp.get("pct_mca_drop", 0.0), 2),
                "Δ Drop (%)": round(dl.get("pct_mca_drop", 0.0), 2),
                "mca_nadir (cm/s)": round(mp.get("mca_nadir_cms", 0.0), 2),
                "Δ Nadir (cm/s)": round(dl.get("mca_nadir_cms", 0.0), 2),
                "recovery_time (s)": round(mp.get("recovery_time_s", 0.0), 2),
                "Δ Rec Time (s)": round(dl.get("recovery_time_s", 0.0), 2),
                "deficit_auc": round(mp.get("cumulative_deficit_auc", 0.0), 2),
                "Δ Deficit AUC": round(dl.get("cumulative_deficit_auc", 0.0), 2),
            })

        return pd.DataFrame(rows)

    def sensitivity_analysis(
        self,
        base_features: pd.Series,
        feature_name: str,
        value_range: Union[np.ndarray, List[float]],
    ) -> pd.DataFrame:
        """
        Sweep a single feature across a continuous range to quantify response elasticity.
        
        Args:
            base_features: Original patient feature vector (pd.Series).
            feature_name: Feature column to modulate (e.g., 'hba1c', 'resting_map').
            value_range: Array of values to test.
            
        Returns:
            pd.DataFrame with feature_value and forecasted hemodynamic targets.
        """
        rows: List[Dict[str, Any]] = []

        for val in value_range:
            v = float(val)
            sim_res = self.simulate(base_features, {feature_name: v})
            preds = sim_res["modified_prediction"]

            row = {
                "feature_value": v,
                "pct_mca_drop": round(preds.get("pct_mca_drop", 0.0), 3),
                "mca_nadir_cms": round(preds.get("mca_nadir_cms", 0.0), 3),
                "recovery_time_s": round(preds.get("recovery_time_s", 0.0), 3),
                "cumulative_deficit_auc": round(preds.get("cumulative_deficit_auc", 0.0), 3),
                "time_to_nadir_s": round(preds.get("time_to_nadir_s", 0.0), 3),
                "recovery_status": round(preds.get("recovery_status", 0.0), 3),
            }
            rows.append(row)

        df_sens = pd.DataFrame(rows)
        df_sens.rename(columns={"feature_value": feature_name}, inplace=True)
        return df_sens

    @staticmethod
    def get_preset_scenarios() -> List[Dict[str, Any]]:
        """
        Return preset clinically-relevant physiological what-if scenarios.
        
        Scenarios:
          1. Improved Glycemic Control: HbA1c reduced to 7.0%.
          2. Blood Pressure Optimization: BP lowered to 120/75 mmHg (MAP ~90 mmHg).
          3. Younger (-10 years): Chronological age dynamically reduced by 10 years.
          4. Better Autonomic Function: RMSSD/SDNN elevated, LF/HF balance normalized.
          5. Healthy Control Profile: Non-diabetic, normal metabolic & autonomic status.
        """
        return [
            {
                "name": "Improved Glycemic Control",
                "modifications": {"hba1c": 7.0, "t2d_status": 1.0},
            },
            {
                "name": "Blood Pressure Optimization",
                "modifications": {
                    "resting_sbp": 120.0,
                    "resting_dbp": 75.0,
                    "resting_map": 90.0,
                    "hypertension": 0.0,
                },
            },
            {
                "name": "Younger (-10 years)",
                "modifications": {},  # Dynamically computed in compare_scenarios / simulate
            },
            {
                "name": "Better Autonomic Function",
                "modifications": {
                    "baseline_rmssd": 40.0,
                    "baseline_sdnn": 140.0,
                    "baseline_lf_hf_ratio": 1.2,
                },
            },
            {
                "name": "Healthy Control Profile",
                "modifications": {
                    "t2d_status": 0.0,
                    "hba1c": 5.2,
                    "diabetes_duration": 0.0,
                    "hypertension": 0.0,
                    "bmi": 24.0,
                    "baseline_rmssd": 40.0,
                    "baseline_sdnn": 140.0,
                    "baseline_lf_hf_ratio": 1.2,
                },
            },
        ]
