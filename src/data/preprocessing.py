"""CerebroTwin - Physiological Signal Preprocessing and Artifact Filtering.

Applies robust signal processing to multimodal hemodynamic waveforms:
- Sit-to-stand alignment at t=0
- Spike artifact removal using median filtering
- Butterworth low-pass anti-aliasing / noise filtration
- Outlier boundary enforcement and linear interpolation
- Physiological MAP computation (MAP = DBP + 1/3*(SBP - DBP))
- Uniform time-grid resampling at 50Hz

CRITICAL DOMAIN RULES:
1. ALWAYS use "MCA blood-flow velocity" or "MCAv" (cm/s), NEVER "cerebral blood flow" or volumetric perfusion.
2. MCAv (TCD) is GROUND TRUTH for training ONLY — NEVER a runtime input feature.
3. Abnormality flags: "Research Response Status: HIGH DEVIATION" — NEVER clinical diagnosis.
4. t=0 is the sit-to-stand transition moment.
5. Subject-level splitting only (GroupKFold / LeaveOneGroupOut by subject_id).
6. N=39 subjects: 20 T2D, 19 Controls, aged 50-85.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt


class WaveformPreprocessor:
    """Preprocesses raw or synthetic physiological time-series waveforms."""

    # Physiological plausibility bounds
    PHYSIOLOGICAL_BOUNDS = {
        "mcav_cms": (5.0, 160.0),      # MCA blood-flow velocity in cm/s
        "ecg_hr_bpm": (30.0, 220.0),    # Heart rate in bpm
        "sbp_mmhg": (40.0, 260.0),      # Systolic blood pressure
        "dbp_mmhg": (20.0, 160.0),      # Diastolic blood pressure
        "map_mmhg": (30.0, 190.0),      # Mean arterial pressure
        "resp_rate_bpm": (4.0, 45.0),   # Respiratory rate
        "etco2_mmhg": (15.0, 65.0),     # End-tidal CO2
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize waveform preprocessor.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        data_cfg = self.config.get("data", {})
        self.target_hz: float = float(data_cfg.get("sampling_rate_hz", 50.0))
        self.pre_stand_s: float = float(data_cfg.get("window_pre_stand_s", 60.0))
        self.post_stand_s: float = float(data_cfg.get("window_post_stand_s", 60.0))
        self.lowpass_cutoff_hz: float = 8.0  # Pass physiological dynamics, attenuate sensor noise
        self.filter_order: int = 2

    def compute_map(self, wf: pd.DataFrame) -> pd.DataFrame:
        """Ensure MAP is computed via standard physiological formula: MAP = DBP + 1/3*(SBP - DBP).

        Args:
            wf: Waveform DataFrame containing hemodynamic signals.

        Returns:
            DataFrame with verified 'map_mmhg' column.
        """
        df = wf.copy()
        has_sbp = "sbp_mmhg" in df.columns
        has_dbp = "dbp_mmhg" in df.columns

        if has_sbp and has_dbp:
            computed_map = df["dbp_mmhg"] + (df["sbp_mmhg"] - df["dbp_mmhg"]) / 3.0
            if "map_mmhg" not in df.columns or df["map_mmhg"].isna().any():
                df["map_mmhg"] = computed_map
            else:
                # If MAP exists but has significant discrepancies, reconcile
                discrepancy = np.abs(df["map_mmhg"] - computed_map)
                if np.nanmean(discrepancy) > 5.0:
                    df["map_mmhg"] = computed_map

        return df

    def align_to_stand(self, wf: pd.DataFrame, stand_time_s: Optional[float] = None) -> pd.DataFrame:
        """Align time series such that t=0 represents the sit-to-stand transition.

        Args:
            wf: Waveform DataFrame.
            stand_time_s: Stand transition time in raw seconds. If None, assumes t=0 is
                          already the transition or searches for pre_stand_s offset.

        Returns:
            Aligned DataFrame windowed to [-pre_stand_s, +post_stand_s].
        """
        df = wf.copy()

        if "time_s" not in df.columns:
            # Create synthetic time vector assuming standard sampling
            df["time_s"] = np.arange(len(df)) / self.target_hz - self.pre_stand_s

        if stand_time_s is not None:
            df["time_s"] = df["time_s"] - stand_time_s
        else:
            # If time starts at 0 and spans ~120s, stand occurred at pre_stand_s
            min_t = df["time_s"].min()
            max_t = df["time_s"].max()
            if min_t >= -0.5 and max_t >= (self.pre_stand_s + self.post_stand_s - 1.0):
                df["time_s"] = df["time_s"] - self.pre_stand_s

        # Slice to standard window [-pre_stand_s, post_stand_s]
        tol = 0.05
        mask = (df["time_s"] >= (-self.pre_stand_s - tol)) & (df["time_s"] <= (self.post_stand_s + tol))
        if mask.any():
            df = df.loc[mask].reset_index(drop=True)

        return df

    def filter_signals(self, wf: pd.DataFrame) -> pd.DataFrame:
        """Apply spike removal, outlier clipping, missing value interpolation, and low-pass filtering.

        Args:
            wf: Waveform DataFrame.

        Returns:
            Filtered, cleaned DataFrame.
        """
        df = wf.copy()
        numeric_cols = [c for c in df.columns if c != "time_s" and pd.api.types.is_numeric_dtype(df[c])]

        # 1. Outlier removal and physiological boundary clipping
        for col in numeric_cols:
            if col in self.PHYSIOLOGICAL_BOUNDS:
                low_b, high_b = self.PHYSIOLOGICAL_BOUNDS[col]
                # Flag extreme non-physiological artifacts as NaN
                is_invalid = (df[col] < low_b * 0.5) | (df[col] > high_b * 1.5)
                df.loc[is_invalid, col] = np.nan

            # Interpolate NaNs
            df[col] = df[col].interpolate(method="linear").bfill().ffill()

        # 2. Median filter for impulsive spike artifact rejection
        for col in numeric_cols:
            vals = df[col].to_numpy()
            if len(vals) >= 5:
                # 5-point median filter (0.1s at 50Hz)
                try:
                    df[col] = medfilt(vals, kernel_size=5)
                except Exception:
                    pass

        # 3. Butterworth zero-phase lowpass filter (filtfilt)
        fs = self.target_hz
        nyq = 0.5 * fs
        normal_cutoff = min(self.lowpass_cutoff_hz / nyq, 0.95)

        if len(df) > 15:
            b, a = butter(self.filter_order, normal_cutoff, btype="low", analog=False)
            for col in numeric_cols:
                vals = df[col].to_numpy(dtype=float)
                try:
                    filtered_vals = filtfilt(b, a, vals)
                    # Enforce hard physiological clipping post-filtering
                    if col in self.PHYSIOLOGICAL_BOUNDS:
                        low_b, high_b = self.PHYSIOLOGICAL_BOUNDS[col]
                        filtered_vals = np.clip(filtered_vals, low_b, high_b)
                    df[col] = filtered_vals
                except Exception:
                    pass

        return df

    def resample_uniform(self, wf: pd.DataFrame, target_hz: Optional[float] = None) -> pd.DataFrame:
        """Resample time series onto an exact, uniformly spaced grid at target_hz.

        Args:
            wf: Waveform DataFrame containing 'time_s'.
            target_hz: Target sampling rate. Defaults to self.target_hz.

        Returns:
            Uniformly resampled DataFrame.
        """
        df = wf.copy()
        fs = target_hz or self.target_hz
        total_seconds = self.pre_stand_s + self.post_stand_s
        n_samples = int(round(total_seconds * fs))

        target_times = np.linspace(-self.pre_stand_s, self.post_stand_s, n_samples, endpoint=False)
        target_times = np.round(target_times, 4)

        if "time_s" not in df.columns:
            df["time_s"] = np.linspace(df["time_s"].min(), df["time_s"].max(), len(df))

        # Sort by time to ensure monotonic interpolation
        df = df.sort_values("time_s").drop_duplicates(subset=["time_s"]).reset_index(drop=True)
        raw_times = df["time_s"].to_numpy()

        resampled_dict: Dict[str, np.ndarray] = {"time_s": target_times}
        for col in df.columns:
            if col == "time_s":
                continue
            vals = df[col].to_numpy(dtype=float)
            resampled_dict[col] = np.round(np.interp(target_times, raw_times, vals), 2)

        return pd.DataFrame(resampled_dict)

    def process_single(self, wf: pd.DataFrame, stand_time_s: Optional[float] = None) -> pd.DataFrame:
        """Complete preprocessing pipeline for an individual subject waveform.

        Steps:
        1. Compute/verify MAP
        2. Align stand transition to t=0
        3. Filter signals and clean artifacts
        4. Resample onto strict 50Hz grid (-60.0s to +60.0s)
        """
        df = self.compute_map(wf)
        df = self.align_to_stand(df, stand_time_s=stand_time_s)
        df = self.filter_signals(df)
        df = self.resample_uniform(df)
        return df

    def process(self, waveforms: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess cohort dictionary of subject waveforms.

        Args:
            waveforms: Dictionary mapping subject_id -> raw waveform DataFrame.

        Returns:
            Dictionary mapping subject_id -> cleaned, aligned, resampled DataFrame.
        """
        processed: Dict[str, pd.DataFrame] = {}
        for subj_id, wf in waveforms.items():
            processed[subj_id] = self.process_single(wf)
        return processed
