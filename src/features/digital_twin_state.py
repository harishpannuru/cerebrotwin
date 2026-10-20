"""
CerebroTwin - Digital Twin State Representation
Organizes physiological features across static, resting baseline, and dynamic tiers.

CRITICAL DOMAIN RULES:
- MCA blood-flow velocity (MCAv in cm/s) is GROUND TRUTH ONLY — NEVER a runtime input feature.
- t=0 represents the sit-to-stand transition moment.
- Waveform features strictly utilize peripheral physiology (ECG HR, SBP, DBP, MAP, EtCO2).
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


class DigitalTwinState:
    """
    Represents the complete state of a patient's digital twin, organizing features
    into three tiers:
      1. Static long-term features (demographics, clinical diagnosis, HbA1c, comorbidities)
      2. Resting baseline features (pre-test hemodynamics, autonomic HRV indices)
      3. Dynamic transition features (derived from continuous peripheral waveforms)
    
    MCAv is explicitly excluded from input features to maintain physiological validity
    as a purely non-invasive digital twin forecaster.
    """

    def __init__(
        self,
        subject_meta: Union[Dict[str, Any], pd.Series],
        baseline_waveform: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize DigitalTwinState with subject metadata and optional waveform.
        
        Args:
            subject_meta: Dictionary or pandas Series containing clinical metadata.
            baseline_waveform: Continuous peripheral waveform DataFrame (-60s to +60s).
        """
        self.subject_meta = subject_meta
        self.baseline_waveform = baseline_waveform

    def _get_meta_val(self, key: str, default: Any = 0.0) -> Any:
        """Helper to safely retrieve a key from dict, pd.Series, or object."""
        if isinstance(self.subject_meta, dict):
            val = self.subject_meta.get(key, default)
        elif hasattr(self.subject_meta, "__getitem__"):
            try:
                val = self.subject_meta[key]
            except (KeyError, IndexError):
                val = default
        else:
            val = getattr(self.subject_meta, key, default)

        if val is None or pd.isna(val):
            return default
        return val

    def get_static_features(self) -> Dict[str, float]:
        """
        Long-term patient features: age, sex, T2D status, HbA1c, hypertension, BMI.
        
        Returns:
            Dictionary with keys:
              - age (float)
              - sex_male (1.0 if 'M', else 0.0)
              - t2d_status (1.0 if Type 2 Diabetes, else 0.0)
              - hba1c (% HbA1c)
              - diabetes_duration (years, 0.0 for controls)
              - hypertension (1.0 if present, else 0.0)
              - bmi (kg/m^2)
        """
        # Demographics & Clinical status
        age = float(self._get_meta_val("age", 65.0))
        
        sex_raw = str(self._get_meta_val("sex", "M")).strip().upper()
        sex_male = 1.0 if sex_raw in ["M", "MALE", "1"] else 0.0

        group_raw = str(self._get_meta_val("group", "")).strip().upper()
        t2d_val = self._get_meta_val("t2d_status", None)
        if t2d_val is not None and not pd.isna(t2d_val):
            t2d_status = 1.0 if float(t2d_val) == 1.0 else 0.0
        else:
            t2d_status = 1.0 if group_raw in ["T2D", "DIABETES", "DIABETIC"] else 0.0

        hba1c = float(self._get_meta_val("hba1c", 5.5 if t2d_status == 0.0 else 7.8))

        if t2d_status == 0.0:
            diabetes_duration = 0.0
        else:
            dur = self._get_meta_val("diabetes_duration_years", None)
            if dur is None or pd.isna(dur):
                dur = self._get_meta_val("diabetes_duration", 5.0)
            diabetes_duration = float(dur)

        hypertension = float(self._get_meta_val("hypertension", 0.0))
        bmi = float(self._get_meta_val("bmi", 26.5))

        return {
            "age": age,
            "sex_male": sex_male,
            "t2d_status": t2d_status,
            "hba1c": hba1c,
            "diabetes_duration": diabetes_duration,
            "hypertension": hypertension,
            "bmi": bmi,
        }

    def get_baseline_features(self) -> Dict[str, float]:
        """
        Resting/baseline features extracted from clinical metadata.
        
        Returns:
            Dictionary with keys:
              - resting_hr (bpm)
              - resting_sbp (mmHg)
              - resting_dbp (mmHg)
              - resting_map (mmHg)
              - baseline_rmssd (ms, parasympathetic vagal tone)
              - baseline_sdnn (ms, total HRV power)
              - baseline_lf_hf_ratio (autonomic sympathovagal balance)
              - baseline_resp_rate (breaths/min)
        """
        resting_sbp = float(self._get_meta_val("resting_sbp", 120.0))
        resting_dbp = float(self._get_meta_val("resting_dbp", 78.0))
        default_map = (resting_sbp + 2.0 * resting_dbp) / 3.0
        resting_map = float(self._get_meta_val("resting_map", default_map))
        resting_hr = float(self._get_meta_val("resting_hr", 72.0))

        baseline_rmssd = float(self._get_meta_val("baseline_rmssd", 28.0))
        baseline_sdnn = float(self._get_meta_val("baseline_sdnn", 45.0))
        baseline_lf_hf_ratio = float(self._get_meta_val("baseline_lf_hf_ratio", 1.6))
        baseline_resp_rate = float(self._get_meta_val("baseline_resp_rate", 14.0))

        return {
            "resting_hr": resting_hr,
            "resting_sbp": resting_sbp,
            "resting_dbp": resting_dbp,
            "resting_map": resting_map,
            "baseline_rmssd": baseline_rmssd,
            "baseline_sdnn": baseline_sdnn,
            "baseline_lf_hf_ratio": baseline_lf_hf_ratio,
            "baseline_resp_rate": baseline_resp_rate,
        }

    def get_dynamic_features(
        self,
        current_waveform: Optional[pd.DataFrame] = None,
        t_current: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute dynamic physiological features from peripheral waveform up to t_current.
        
        Pre-stand features (window t = [-60.0, 0.0]):
          pre_stand_hr_mean, pre_stand_hr_std, pre_stand_map_mean, pre_stand_map_std,
          pre_stand_etco2_mean, pre_stand_sbp_mean, pre_stand_dbp_mean
        
        Early transition features (window t = [0.0, min(t_current, 5.0)], if t_current > 0):
          early_map_slope (linear regression slope of MAP),
          early_hr_slope (linear regression slope of HR),
          early_sbp_drop_5s (SBP at t=5 minus SBP at t=0),
          early_etco2_change (EtCO2 at t=5 minus EtCO2 at t=0)
        
        Derived:
          baroreflex_sensitivity_proxy = (HR_change) / (SBP_change) in first 5s.
        """
        wf = current_waveform if current_waveform is not None else self.baseline_waveform

        # Baseline resting defaults
        bl = self.get_baseline_features()
        default_hr = bl["resting_hr"]
        default_map = bl["resting_map"]
        default_sbp = bl["resting_sbp"]
        default_dbp = bl["resting_dbp"]
        default_etco2 = 38.0

        if wf is None or len(wf) == 0 or "time_s" not in wf.columns:
            return {
                "pre_stand_hr_mean": default_hr,
                "pre_stand_hr_std": 0.0,
                "pre_stand_map_mean": default_map,
                "pre_stand_map_std": 0.0,
                "pre_stand_etco2_mean": default_etco2,
                "pre_stand_sbp_mean": default_sbp,
                "pre_stand_dbp_mean": default_dbp,
                "early_map_slope": 0.0,
                "early_hr_slope": 0.0,
                "early_sbp_drop_5s": 0.0,
                "early_etco2_change": 0.0,
                "baroreflex_sensitivity_proxy": 0.0,
            }

        # 1. Pre-stand features (t in [-60.0, 0.0])
        pre_stand_wf = wf[(wf["time_s"] >= -60.0) & (wf["time_s"] <= 0.0)]
        if len(pre_stand_wf) > 0:
            hr_col = "ecg_hr_bpm" if "ecg_hr_bpm" in pre_stand_wf.columns else ("hr" if "hr" in pre_stand_wf.columns else None)
            map_col = "map_mmhg" if "map_mmhg" in pre_stand_wf.columns else ("map" if "map" in pre_stand_wf.columns else None)
            sbp_col = "sbp_mmhg" if "sbp_mmhg" in pre_stand_wf.columns else ("sbp" if "sbp" in pre_stand_wf.columns else None)
            dbp_col = "dbp_mmhg" if "dbp_mmhg" in pre_stand_wf.columns else ("dbp" if "dbp" in pre_stand_wf.columns else None)
            etco2_col = "etco2_mmhg" if "etco2_mmhg" in pre_stand_wf.columns else ("etco2" if "etco2" in pre_stand_wf.columns else None)

            pre_hr_mean = float(pre_stand_wf[hr_col].mean()) if hr_col else default_hr
            pre_hr_std = float(pre_stand_wf[hr_col].std(ddof=1)) if hr_col and len(pre_stand_wf) > 1 else 0.0
            pre_map_mean = float(pre_stand_wf[map_col].mean()) if map_col else default_map
            pre_map_std = float(pre_stand_wf[map_col].std(ddof=1)) if map_col and len(pre_stand_wf) > 1 else 0.0
            pre_sbp_mean = float(pre_stand_wf[sbp_col].mean()) if sbp_col else default_sbp
            pre_dbp_mean = float(pre_stand_wf[dbp_col].mean()) if dbp_col else default_dbp
            pre_etco2_mean = float(pre_stand_wf[etco2_col].mean()) if etco2_col else default_etco2

            if np.isnan(pre_hr_std):
                pre_hr_std = 0.0
            if np.isnan(pre_map_std):
                pre_map_std = 0.0
        else:
            pre_hr_mean, pre_hr_std = default_hr, 0.0
            pre_map_mean, pre_map_std = default_map, 0.0
            pre_sbp_mean, pre_dbp_mean = default_sbp, default_dbp
            pre_etco2_mean = default_etco2

        # 2. Early transition features (from t=0 to t_current, if t_current > 0)
        early_map_slope = 0.0
        early_hr_slope = 0.0
        early_sbp_drop_5s = 0.0
        early_etco2_change = 0.0
        baroreflex_sensitivity_proxy = 0.0

        if t_current > 0.0:
            t_max = min(float(t_current), 5.0)
            trans_wf = wf[(wf["time_s"] >= 0.0) & (wf["time_s"] <= t_max)].sort_values("time_s")

            if len(trans_wf) >= 2:
                t_arr = trans_wf["time_s"].values
                hr_col = "ecg_hr_bpm" if "ecg_hr_bpm" in trans_wf.columns else ("hr" if "hr" in trans_wf.columns else None)
                map_col = "map_mmhg" if "map_mmhg" in trans_wf.columns else ("map" if "map" in trans_wf.columns else None)
                sbp_col = "sbp_mmhg" if "sbp_mmhg" in trans_wf.columns else ("sbp" if "sbp" in trans_wf.columns else None)
                etco2_col = "etco2_mmhg" if "etco2_mmhg" in trans_wf.columns else ("etco2" if "etco2" in trans_wf.columns else None)

                # Linear regression slopes
                if map_col:
                    early_map_slope = float(np.polyfit(t_arr, trans_wf[map_col].values, deg=1)[0])
                if hr_col:
                    early_hr_slope = float(np.polyfit(t_arr, trans_wf[hr_col].values, deg=1)[0])

                # Changes at end of early window minus t=0
                if sbp_col:
                    sbp_t_start = float(trans_wf[sbp_col].iloc[0])
                    sbp_t_end = float(trans_wf[sbp_col].iloc[-1])
                    early_sbp_drop_5s = sbp_t_end - sbp_t_start

                if etco2_col:
                    etco2_t_start = float(trans_wf[etco2_col].iloc[0])
                    etco2_t_end = float(trans_wf[etco2_col].iloc[-1])
                    early_etco2_change = etco2_t_end - etco2_t_start

                # Baroreflex proxy: (HR_change) / (SBP_change)
                if hr_col:
                    hr_change = float(trans_wf[hr_col].iloc[-1] - trans_wf[hr_col].iloc[0])
                else:
                    hr_change = 0.0

                denom = early_sbp_drop_5s
                if abs(denom) < 1e-4:
                    denom = 1e-4 if denom >= 0 else -1e-4
                baroreflex_sensitivity_proxy = float(hr_change / denom)

        return {
            "pre_stand_hr_mean": pre_hr_mean,
            "pre_stand_hr_std": pre_hr_std,
            "pre_stand_map_mean": pre_map_mean,
            "pre_stand_map_std": pre_map_std,
            "pre_stand_etco2_mean": pre_etco2_mean,
            "pre_stand_sbp_mean": pre_sbp_mean,
            "pre_stand_dbp_mean": pre_dbp_mean,
            "early_map_slope": early_map_slope,
            "early_hr_slope": early_hr_slope,
            "early_sbp_drop_5s": early_sbp_drop_5s,
            "early_etco2_change": early_etco2_change,
            "baroreflex_sensitivity_proxy": baroreflex_sensitivity_proxy,
        }

    def get_all_features(
        self,
        current_waveform: Optional[pd.DataFrame] = None,
        t_current: float = 0.0
    ) -> Dict[str, float]:
        """
        Merge static, resting baseline, and dynamic features into one flat dictionary.
        
        Guaranteed to output all 27 feature columns in consistent order without leakage.
        """
        features: Dict[str, float] = {}
        features.update(self.get_static_features())
        features.update(self.get_baseline_features())
        features.update(self.get_dynamic_features(current_waveform=current_waveform, t_current=t_current))
        return features

    def to_dataframe_row(
        self,
        current_waveform: Optional[pd.DataFrame] = None,
        t_current: float = 0.0
    ) -> pd.DataFrame:
        """
        Return all patient features as a single-row DataFrame.
        Uses subject_id as the index if available.
        """
        features = self.get_all_features(current_waveform=current_waveform, t_current=t_current)
        subject_id = self._get_meta_val("subject_id", "subject_00")
        return pd.DataFrame([features], index=[subject_id])
