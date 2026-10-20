"""
CerebroTwin - Cohort Feature Engineering and Ground Truth Target Extraction

CRITICAL DOMAIN RULES:
1. ALWAYS use "MCA blood-flow velocity" or "MCAv" (cm/s), NEVER "cerebral blood flow" or volumetric perfusion.
2. MCA blood-flow velocity (MCAv from TCD) is GROUND TRUTH for training ONLY — NEVER a runtime input feature.
3. Abnormality flags: "Research Response Status: HIGH DEVIATION" — NEVER clinical diagnosis.
4. t=0 is the sit-to-stand transition moment. Continuous waveforms span -60s to +60s.
5. Cohort structure: N=39 subjects (20 T2D, 19 Controls, aged 50-85).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .digital_twin_state import DigitalTwinState


def get_feature_names() -> List[str]:
    """
    Return the standard ordered list of all 27 feature column names.
    Organized into:
      - Static clinical & demographic features (7)
      - Resting baseline features (8)
      - Pre-stand continuous peripheral dynamic features (7)
      - Early transition peripheral dynamic features (5)
    """
    return [
        "age",
        "sex_male",
        "t2d_status",
        "hba1c",
        "diabetes_duration",
        "hypertension",
        "bmi",
        "resting_hr",
        "resting_sbp",
        "resting_dbp",
        "resting_map",
        "baseline_rmssd",
        "baseline_sdnn",
        "baseline_lf_hf_ratio",
        "baseline_resp_rate",
        "pre_stand_hr_mean",
        "pre_stand_hr_std",
        "pre_stand_map_mean",
        "pre_stand_map_std",
        "pre_stand_etco2_mean",
        "pre_stand_sbp_mean",
        "pre_stand_dbp_mean",
        "early_map_slope",
        "early_hr_slope",
        "early_sbp_drop_5s",
        "early_etco2_change",
        "baroreflex_sensitivity_proxy",
    ]


def get_target_names() -> List[str]:
    """
    Return ordered list of primary target column names derived from MCA blood-flow velocity.
    """
    return [
        "pct_mca_drop",
        "mca_nadir_cms",
        "time_to_nadir_s",
        "cumulative_deficit_auc",
        "recovery_time_s",
        "recovery_status",
    ]


def get_trajectory_target_names() -> List[str]:
    """
    Return trajectory knot column names representing MCAv (% of pre-stand baseline)
    at 5-second intervals from t=0s to t=60s (13 knots total).
    """
    return [f"traj_t{t}" for t in range(0, 65, 5)]


def compute_pre_stand_features(wf: pd.DataFrame) -> Dict[str, float]:
    """
    Compute features from the pre-stand resting window (t = -60s to t = 0s).
    Only peripheral signals (ECG HR, MAP, EtCO2, SBP, DBP) are used.
    
    Args:
        wf: Continuous peripheral waveform DataFrame containing 'time_s'.
        
    Returns:
        Dictionary of pre-stand statistics (means and standard deviations).
    """
    default_vals = {
        "pre_stand_hr_mean": 70.0,
        "pre_stand_hr_std": 0.0,
        "pre_stand_map_mean": 93.0,
        "pre_stand_map_std": 0.0,
        "pre_stand_etco2_mean": 38.0,
        "pre_stand_sbp_mean": 120.0,
        "pre_stand_dbp_mean": 80.0,
    }

    if wf is None or len(wf) == 0 or "time_s" not in wf.columns:
        return default_vals

    pre_mask = (wf["time_s"] >= -60.0) & (wf["time_s"] <= 0.0)
    pre_wf = wf[pre_mask]

    if len(pre_wf) == 0:
        return default_vals

    hr_col = "ecg_hr_bpm" if "ecg_hr_bpm" in pre_wf.columns else ("hr" if "hr" in pre_wf.columns else None)
    map_col = "map_mmhg" if "map_mmhg" in pre_wf.columns else ("map" if "map" in pre_wf.columns else None)
    sbp_col = "sbp_mmhg" if "sbp_mmhg" in pre_wf.columns else ("sbp" if "sbp" in pre_wf.columns else None)
    dbp_col = "dbp_mmhg" if "dbp_mmhg" in pre_wf.columns else ("dbp" if "dbp" in pre_wf.columns else None)
    etco2_col = "etco2_mmhg" if "etco2_mmhg" in pre_wf.columns else ("etco2" if "etco2" in pre_wf.columns else None)

    hr_m = float(pre_wf[hr_col].mean()) if hr_col else default_vals["pre_stand_hr_mean"]
    hr_s = float(pre_wf[hr_col].std(ddof=1)) if hr_col and len(pre_wf) > 1 else 0.0
    map_m = float(pre_wf[map_col].mean()) if map_col else default_vals["pre_stand_map_mean"]
    map_s = float(pre_wf[map_col].std(ddof=1)) if map_col and len(pre_wf) > 1 else 0.0
    sbp_m = float(pre_wf[sbp_col].mean()) if sbp_col else default_vals["pre_stand_sbp_mean"]
    dbp_m = float(pre_wf[dbp_col].mean()) if dbp_col else default_vals["pre_stand_dbp_mean"]
    etco2_m = float(pre_wf[etco2_col].mean()) if etco2_col else default_vals["pre_stand_etco2_mean"]

    return {
        "pre_stand_hr_mean": hr_m,
        "pre_stand_hr_std": 0.0 if np.isnan(hr_s) else hr_s,
        "pre_stand_map_mean": map_m,
        "pre_stand_map_std": 0.0 if np.isnan(map_s) else map_s,
        "pre_stand_etco2_mean": etco2_m,
        "pre_stand_sbp_mean": sbp_m,
        "pre_stand_dbp_mean": dbp_m,
    }


def compute_early_transition_features(wf: pd.DataFrame, t_max: float = 5.0) -> Dict[str, float]:
    """
    Compute peripheral dynamic features from the early sit-to-stand transition (t=0 to t_max).
    
    Args:
        wf: Continuous peripheral waveform DataFrame.
        t_max: Upper boundary in seconds (typically 5.0s).
        
    Returns:
        Dictionary containing early transition slopes and changes.
    """
    defaults = {
        "early_map_slope": 0.0,
        "early_hr_slope": 0.0,
        "early_sbp_drop_5s": 0.0,
        "early_etco2_change": 0.0,
        "baroreflex_sensitivity_proxy": 0.0,
    }

    if wf is None or len(wf) == 0 or "time_s" not in wf.columns or t_max <= 0.0:
        return defaults

    mask = (wf["time_s"] >= 0.0) & (wf["time_s"] <= t_max)
    trans_wf = wf[mask].sort_values("time_s")

    if len(trans_wf) < 2:
        return defaults

    t_arr = trans_wf["time_s"].values
    hr_col = "ecg_hr_bpm" if "ecg_hr_bpm" in trans_wf.columns else ("hr" if "hr" in trans_wf.columns else None)
    map_col = "map_mmhg" if "map_mmhg" in trans_wf.columns else ("map" if "map" in trans_wf.columns else None)
    sbp_col = "sbp_mmhg" if "sbp_mmhg" in trans_wf.columns else ("sbp" if "sbp" in trans_wf.columns else None)
    etco2_col = "etco2_mmhg" if "etco2_mmhg" in trans_wf.columns else ("etco2" if "etco2" in trans_wf.columns else None)

    map_slope = float(np.polyfit(t_arr, trans_wf[map_col].values, deg=1)[0]) if map_col else 0.0
    hr_slope = float(np.polyfit(t_arr, trans_wf[hr_col].values, deg=1)[0]) if hr_col else 0.0

    sbp_drop = 0.0
    if sbp_col:
        sbp_drop = float(trans_wf[sbp_col].iloc[-1] - trans_wf[sbp_col].iloc[0])

    etco2_change = 0.0
    if etco2_col:
        etco2_change = float(trans_wf[etco2_col].iloc[-1] - trans_wf[etco2_col].iloc[0])

    hr_change = float(trans_wf[hr_col].iloc[-1] - trans_wf[hr_col].iloc[0]) if hr_col else 0.0
    denom = sbp_drop if abs(sbp_drop) >= 1e-4 else (1e-4 if sbp_drop >= 0 else -1e-4)
    brs_proxy = float(hr_change / denom)

    return {
        "early_map_slope": map_slope,
        "early_hr_slope": hr_slope,
        "early_sbp_drop_5s": sbp_drop,
        "early_etco2_change": etco2_change,
        "baroreflex_sensitivity_proxy": brs_proxy,
    }


def compute_target_metrics(wf: pd.DataFrame, baseline_mcav: Optional[float] = None) -> Dict[str, float]:
    """
    Compute all target metrics from MCA blood-flow velocity (MCAv in cm/s) ground truth.
    
    Target metrics computed:
      - pct_mca_drop: max % drop from pre-stand baseline MCAv
      - mca_nadir_cms: minimum MCA blood-flow velocity value post-stand (cm/s)
      - time_to_nadir_s: time from t=0 to nadir (s)
      - cumulative_deficit_auc: integral of (baseline - MCAv) from t=0 to t=60, normalized
      - recovery_time_s: time to return within 5% of baseline (capped at 60s if no recovery)
      - recovery_status: 0 (Normal, <20s) or 1 (Research Response Status: HIGH DEVIATION, >=20s)
      - traj_t0 ... traj_t60: MCA blood-flow velocity as % of baseline at 5s knots
      
    Args:
        wf: Continuous waveform DataFrame containing 'time_s' and 'mcav_cms'.
        baseline_mcav: Pre-calculated baseline MCAv (cm/s). If None, calculated from [-60, 0]s.
        
    Returns:
        Dictionary mapping target names to computed values.
    """
    # 1. Determine pre-stand baseline MCA blood-flow velocity
    if baseline_mcav is None or baseline_mcav <= 0.0:
        if wf is not None and "time_s" in wf.columns and "mcav_cms" in wf.columns:
            pre_mask = (wf["time_s"] >= -60.0) & (wf["time_s"] <= 0.0)
            pre_mcav = wf.loc[pre_mask, "mcav_cms"]
            if len(pre_mcav) > 0:
                baseline_mcav = float(pre_mcav.mean())
            else:
                baseline_mcav = 60.0
        else:
            baseline_mcav = 60.0

    trajectory_knots = list(range(0, 65, 5))
    default_targets: Dict[str, float] = {
        "pct_mca_drop": 0.0,
        "mca_nadir_cms": baseline_mcav,
        "time_to_nadir_s": 0.0,
        "cumulative_deficit_auc": 0.0,
        "recovery_time_s": 0.0,
        "recovery_status": 0.0,
    }
    for knot in trajectory_knots:
        default_targets[f"traj_t{knot}"] = 100.0

    if wf is None or len(wf) == 0 or "time_s" not in wf.columns or "mcav_cms" not in wf.columns:
        return default_targets

    # 2. Extract post-stand window (t in [0.0, 60.0])
    post_mask = (wf["time_s"] >= 0.0) & (wf["time_s"] <= 60.0)
    post_wf = wf[post_mask].sort_values("time_s").copy()

    if len(post_wf) < 2:
        return default_targets

    t_post = post_wf["time_s"].values
    mcav_post = post_wf["mcav_cms"].values

    # Nadir and drop
    min_idx = int(np.argmin(mcav_post))
    mca_nadir_cms = float(mcav_post[min_idx])
    time_to_nadir_s = float(t_post[min_idx])
    
    pct_mca_drop = float(max(0.0, (baseline_mcav - mca_nadir_cms) / baseline_mcav * 100.0))

    deficit = np.maximum(0.0, baseline_mcav - mcav_post)
    total_time = max(t_post[-1] - t_post[0], 1.0)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    raw_auc = float(_trapz(deficit, t_post))
    cumulative_deficit_auc = float(raw_auc / total_time)

    # Recovery time: time from t=0 to return within 5% of baseline (MCAv >= 0.95 * baseline_mcav)
    recovery_threshold = 0.95 * baseline_mcav
    # Recovery occurs after reaching nadir
    recovery_mask = (t_post >= time_to_nadir_s) & (mcav_post >= recovery_threshold)
    recovery_times = t_post[recovery_mask]

    if len(recovery_times) > 0:
        recovery_time_s = float(min(60.0, recovery_times[0]))
    else:
        recovery_time_s = 60.0

    # Recovery status flag: 0 (Normal, <20s) or 1 (Research Response Status: HIGH DEVIATION, >=20s)
    recovery_status = 1.0 if recovery_time_s >= 20.0 else 0.0

    # Trajectory knot values as % of baseline MCAv
    traj_interp = np.interp(trajectory_knots, t_post, mcav_post)
    traj_pct = (traj_interp / baseline_mcav) * 100.0

    results: Dict[str, float] = {
        "pct_mca_drop": pct_mca_drop,
        "mca_nadir_cms": mca_nadir_cms,
        "time_to_nadir_s": time_to_nadir_s,
        "cumulative_deficit_auc": cumulative_deficit_auc,
        "recovery_time_s": recovery_time_s,
        "recovery_status": recovery_status,
    }
    for knot, val in zip(trajectory_knots, traj_pct):
        results[f"traj_t{knot}"] = float(val)

    return results


def extract_features_for_cohort(
    subjects_df: pd.DataFrame,
    waveforms: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
    t_current: float = 0.0,
) -> pd.DataFrame:
    """
    Extract standardized feature matrix for all subjects in the cohort.
    
    Guarantees:
      - Subject ID indexed.
      - Exactly 27 columns matching get_feature_names().
      - Zero target leakage (no MCAv information in feature matrix).
      
    Args:
        subjects_df: Metadata DataFrame with subject_id column or index.
        waveforms: Dictionary mapping subject_id to continuous waveform DataFrame.
        config: Optional configuration dictionary.
        t_current: Observation threshold in seconds relative to stand (default: 0.0).
        
    Returns:
        pd.DataFrame of features, index=subject_id, columns=get_feature_names().
    """
    rows: List[pd.DataFrame] = []
    feature_cols = get_feature_names()

    for idx, row in subjects_df.iterrows():
        sid = str(row["subject_id"]) if "subject_id" in row else str(idx)
        wf = waveforms.get(sid, None)
        
        # Instantiate DigitalTwinState
        twin_state = DigitalTwinState(subject_meta=row, baseline_waveform=wf)
        feat_row = twin_state.to_dataframe_row(current_waveform=wf, t_current=t_current)
        feat_row.index = [sid]
        
        # Ensure all columns present and properly aligned
        for col in feature_cols:
            if col not in feat_row.columns:
                feat_row[col] = 0.0
        feat_row = feat_row[feature_cols]
        rows.append(feat_row)

    if not rows:
        return pd.DataFrame(columns=feature_cols)

    df_features = pd.concat(rows, axis=0)
    df_features.index.name = "subject_id"
    return df_features


def extract_targets_for_cohort(
    subjects_df: pd.DataFrame,
    waveforms: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Extract ground truth target variables from MCA blood-flow velocity for all subjects.
    
    Target columns:
      - Primary metrics: pct_mca_drop, mca_nadir_cms, time_to_nadir_s,
                         cumulative_deficit_auc, recovery_time_s, recovery_status
      - Continuous trajectory knots: traj_t0 through traj_t60
      
    Args:
        subjects_df: Metadata DataFrame with subject_id column or index.
        waveforms: Dictionary mapping subject_id to continuous waveform DataFrame.
        config: Optional configuration dictionary.
        
    Returns:
        pd.DataFrame of targets with subject_id as index.
    """
    target_cols = get_target_names() + get_trajectory_target_names()
    rows: List[pd.Series] = []
    subject_ids: List[str] = []

    for idx, row in subjects_df.iterrows():
        sid = str(row["subject_id"]) if "subject_id" in row else str(idx)
        wf = waveforms.get(sid, None)

        baseline_mcav = None
        if wf is not None and "time_s" in wf.columns and "mcav_cms" in wf.columns:
            pre_mask = (wf["time_s"] >= -60.0) & (wf["time_s"] <= 0.0)
            pre_mcav = wf.loc[pre_mask, "mcav_cms"]
            if len(pre_mcav) > 0:
                baseline_mcav = float(pre_mcav.mean())

        targets = compute_target_metrics(wf=wf, baseline_mcav=baseline_mcav)
        series = pd.Series(targets, name=sid)
        rows.append(series)
        subject_ids.append(sid)

    if not rows:
        return pd.DataFrame(columns=target_cols)

    df_targets = pd.DataFrame(rows, index=subject_ids)[target_cols]
    df_targets.index.name = "subject_id"
    return df_targets
