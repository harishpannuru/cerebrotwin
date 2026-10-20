"""CerebroTwin - Synthetic Cohort and Physiological Waveform Generator.

Generates realistic multimodal physiological data for N=39 subjects (20 T2D, 19 Controls)
undergoing a 120-second sit-to-stand orthostatic protocol (-60s to +60s at 50Hz).

CRITICAL DOMAIN RULES:
1. ALWAYS use "MCA blood-flow velocity" or "MCAv" (cm/s), NEVER "cerebral blood flow" or volumetric perfusion.
2. MCAv (TCD) is GROUND TRUTH for training ONLY — NEVER a runtime input feature.
3. Abnormality flags: "Research Response Status: HIGH DEVIATION" — NEVER clinical diagnosis.
4. t=0 is the sit-to-stand transition moment.
5. Subject-level splitting only (GroupKFold / LeaveOneGroupOut by subject_id).
6. N=39 subjects: 20 T2D, 19 Controls, aged 50-85.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SyntheticCohortGenerator:
    """Generates synthetic physiological cohorts with realistic orthostatic dynamics.

    Simulates continuous arterial blood pressure (SBP, DBP, MAP), heart rate,
    respiratory rate, end-tidal CO2 (EtCO2), and middle cerebral artery blood-flow
    velocity (MCAv) during a sit-to-stand challenge. Models impaired dynamic cerebral
    autoregulation (dCA) and blunted baroreflex sensitivity in Type 2 Diabetes (T2D).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: int = 42) -> None:
        """Initialize synthetic cohort generator.

        Args:
            config: Optional configuration dictionary. If None, default settings are used.
            seed: Master random seed for reproducible cohort synthesis.
        """
        self.config = config or {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Cohort size configuration
        data_cfg = self.config.get("data", {})
        self.n_t2d: int = data_cfg.get("n_t2d", 20)
        self.n_control: int = data_cfg.get("n_control", 19)
        self.n_subjects: int = self.n_t2d + self.n_control
        self.fs: float = float(data_cfg.get("sampling_rate_hz", 50))
        self.pre_stand_s: float = float(data_cfg.get("window_pre_stand_s", 60))
        self.post_stand_s: float = float(data_cfg.get("window_post_stand_s", 60))

        # Physiological parameter ranges from config or physiological literature
        phys_cfg = self.config.get("physiology", {})
        self.mcav_range: Tuple[float, float] = tuple(phys_cfg.get("mcav_baseline_range", [45.0, 75.0]))
        self.t2d_drop_pct: Tuple[float, float] = tuple(phys_cfg.get("t2d_mcav_drop_pct", [15.0, 40.0]))
        self.ctl_drop_pct: Tuple[float, float] = tuple(phys_cfg.get("control_mcav_drop_pct", [5.0, 20.0]))
        self.hr_range: Tuple[float, float] = tuple(phys_cfg.get("hr_baseline_range", [60.0, 90.0]))
        self.sbp_range: Tuple[float, float] = tuple(phys_cfg.get("sbp_baseline_range", [110.0, 160.0]))
        self.dbp_range: Tuple[float, float] = tuple(phys_cfg.get("dbp_baseline_range", [65.0, 95.0]))
        self.rr_range: Tuple[float, float] = tuple(phys_cfg.get("resp_rate_range", [12.0, 20.0]))
        self.etco2_range: Tuple[float, float] = tuple(phys_cfg.get("etco2_range", [35.0, 45.0]))

        # Cache variables
        self.metadata_df: Optional[pd.DataFrame] = None
        self.waveforms: Optional[Dict[str, pd.DataFrame]] = None

    def _sample_subject_metadata(self, subject_idx: int, group: str) -> Dict[str, Any]:
        """Generate static demographic, clinical, and autonomic baseline features."""
        subject_id = f"S{subject_idx:04d}"
        age = int(self.rng.integers(50, 86))
        sex = "M" if self.rng.random() < 0.52 else "F"

        if group == "T2D":
            hba1c = round(float(self.rng.uniform(7.0, 11.0)), 1)
            diabetes_duration_years = round(float(self.rng.uniform(2.0, 25.0)), 1)
            hypertension = bool(self.rng.random() < 0.65)
            bmi = round(float(self.rng.uniform(28.0, 42.0)), 1)

            # Autonomic metrics (reduced HRV in T2D)
            baseline_rmssd = round(float(self.rng.uniform(15.0, 30.0)), 1)
            baseline_sdnn = round(float(self.rng.uniform(80.0, 120.0)), 1)
            baseline_lf_hf_ratio = round(float(self.rng.uniform(1.5, 4.0)), 2)

            # Medications typical for T2D
            med_pool = ["metformin", "ACE_inhibitor", "statin", "SGLT2_inhibitor", "sulfonylurea", "beta_blocker"]
            med_count = int(self.rng.integers(2, 5))
            medications = sorted(list(self.rng.choice(med_pool, size=med_count, replace=False)))

            # Comorbidities
            comorb_pool = ["peripheral_neuropathy", "retinopathy", "dyslipidemia", "nephropathy"]
            comorb_count = int(self.rng.integers(1, 3))
            comorbidities = sorted(list(self.rng.choice(comorb_pool, size=comorb_count, replace=False)))
        else:
            hba1c = round(float(self.rng.uniform(4.5, 5.9)), 1)
            diabetes_duration_years = None
            hypertension = bool(self.rng.random() < 0.30)
            bmi = round(float(self.rng.uniform(22.0, 30.0)), 1)

            # Preserved HRV in Controls
            baseline_rmssd = round(float(self.rng.uniform(25.0, 55.0)), 1)
            baseline_sdnn = round(float(self.rng.uniform(100.0, 160.0)), 1)
            baseline_lf_hf_ratio = round(float(self.rng.uniform(0.8, 2.5)), 2)

            if hypertension:
                med_pool = ["ACE_inhibitor", "statin", "thiazide_diuretic"]
                med_count = int(self.rng.integers(1, 3))
                medications = sorted(list(self.rng.choice(med_pool, size=med_count, replace=False)))
            else:
                medications = ["statin"] if self.rng.random() < 0.25 else []
            comorbidities = ["dyslipidemia"] if self.rng.random() < 0.20 else []

        # Hemodynamic baseline resting values
        resting_hr = round(float(self.rng.uniform(self.hr_range[0], self.hr_range[1])), 1)
        resting_sbp = round(float(self.rng.uniform(self.sbp_range[0], self.sbp_range[1])), 1)
        # SBP slightly higher in hypertensive
        if hypertension:
            resting_sbp = min(175.0, round(resting_sbp + float(self.rng.uniform(10.0, 20.0)), 1))

        resting_dbp = round(float(self.rng.uniform(self.dbp_range[0], self.dbp_range[1])), 1)
        if resting_dbp >= resting_sbp - 20:
            resting_dbp = resting_sbp - 35.0

        resting_map = round(float(resting_dbp + (resting_sbp - resting_dbp) / 3.0), 1)
        baseline_resp_rate = round(float(self.rng.uniform(self.rr_range[0], self.rr_range[1])), 1)
        baseline_etco2 = round(float(self.rng.uniform(self.etco2_range[0], self.etco2_range[1])), 1)
        resting_mcav = round(float(self.rng.uniform(self.mcav_range[0], self.mcav_range[1])), 1)

        return {
            "subject_id": subject_id,
            "group": group,
            "age": age,
            "sex": sex,
            "hba1c": hba1c,
            "diabetes_duration_years": diabetes_duration_years,
            "hypertension": hypertension,
            "bmi": bmi,
            "resting_hr": resting_hr,
            "resting_sbp": resting_sbp,
            "resting_dbp": resting_dbp,
            "resting_map": resting_map,
            "resting_mcav_cms": resting_mcav,
            "baseline_resp_rate": baseline_resp_rate,
            "baseline_etco2_mmhg": baseline_etco2,
            "baseline_rmssd": baseline_rmssd,
            "baseline_sdnn": baseline_sdnn,
            "baseline_lf_hf_ratio": baseline_lf_hf_ratio,
            "medications": medications,
            "comorbidities": comorbidities,
        }

    def _generate_subject_waveforms(self, meta: Dict[str, Any]) -> pd.DataFrame:
        """Synthesize 120-second continuous multimodal physiological waveforms at 50Hz.

        Models:
        - Sitting baseline with respiratory sinus arrhythmia (RSA) and Mayer waves.
        - Transition at t=0 (sit-to-stand).
        - Post-stand transient orthostatic drop, nadir, and recovery.
        - Differential dynamic cerebral autoregulation (dCA) between T2D and Control.
        """
        dt = 1.0 / self.fs
        # Time vector: exactly 6000 points from -60.0s to +59.98s
        total_seconds = self.pre_stand_s + self.post_stand_s
        n_samples = int(round(total_seconds * self.fs))
        t = np.linspace(-self.pre_stand_s, self.post_stand_s, n_samples, endpoint=False)
        t_post = np.maximum(0.0, t)
        is_post = (t >= 0.0).astype(float)

        is_t2d = meta["group"] == "T2D"
        hr_base = float(meta["resting_hr"])
        sbp_base = float(meta["resting_sbp"])
        dbp_base = float(meta["resting_dbp"])
        map_base = float(meta["resting_map"])
        mcav_base = float(meta["resting_mcav_cms"])
        rr_base = float(meta["baseline_resp_rate"])
        etco2_base = float(meta["baseline_etco2_mmhg"])

        # Physiological oscillations
        f_resp = rr_base / 60.0  # respiratory frequency (Hz)
        f_mayer = 0.10 + float(self.rng.normal(0.0, 0.01))  # Mayer wave (~0.1 Hz)
        resp_phase = float(self.rng.uniform(0, 2 * np.pi))
        mayer_phase = float(self.rng.uniform(0, 2 * np.pi))

        # Modulations
        resp_mod = np.sin(2 * np.pi * f_resp * t + resp_phase)
        mayer_mod = np.sin(2 * np.pi * f_mayer * t + mayer_phase)

        # -------------------------------------------------------------
        # 1. Respiratory Rate (bpm) and EtCO2 (mmHg)
        # -------------------------------------------------------------
        rr_delta = float(self.rng.uniform(1.0, 4.0))
        rr_trans = 1.0 / (1.0 + np.exp(-(t - 4.0) / 3.0)) * is_post
        rr_signal = rr_base + rr_delta * rr_trans + 0.3 * resp_mod + self.rng.normal(0, 0.15, n_samples)

        etco2_drop = float(self.rng.uniform(1.5, 4.5))
        etco2_trans = 1.0 / (1.0 + np.exp(-(t - 10.0) / 6.0)) * is_post
        etco2_signal = etco2_base - etco2_drop * etco2_trans + 0.4 * resp_mod + self.rng.normal(0, 0.2, n_samples)

        # -------------------------------------------------------------
        # 2. Blood Pressure Dynamics (MAP, SBP, DBP)
        # -------------------------------------------------------------
        if is_t2d:
            map_drop_mag = float(self.rng.uniform(15.0, 35.0))
            t_nadir_bp = float(self.rng.uniform(7.0, 14.0))
            tau_rec_bp = float(self.rng.uniform(22.0, 45.0))
            sustained_deficit = float(self.rng.uniform(4.0, 10.0))
        else:
            map_drop_mag = float(self.rng.uniform(6.0, 16.0))
            t_nadir_bp = float(self.rng.uniform(5.0, 9.0))
            tau_rec_bp = float(self.rng.uniform(12.0, 20.0))
            sustained_deficit = float(self.rng.uniform(0.0, 4.0))

        # Orthostatic BP drop and baroreflex recovery curve
        # Normalized asymmetric impulse curve reaching 1.0 at t_nadir
        bp_drop_curve = (
            (t_post / t_nadir_bp) * np.exp(1.0 - (t_post / t_nadir_bp))
        )
        bp_recovery_curve = 1.0 - np.exp(-t_post / max(tau_rec_bp, 1e-3))
        map_transient = -map_drop_mag * bp_drop_curve * is_post
        map_sustained = -sustained_deficit * bp_recovery_curve * is_post

        # MAP beat-to-beat variability (Mayer waves + RSA coupling + white noise)
        map_var = 1.8 * mayer_mod + 0.8 * resp_mod + self.rng.normal(0, 0.7, n_samples)
        map_signal = map_base + map_transient + map_sustained + map_var

        # Pulse pressure dynamics: PP transiently narrows during nadir
        pp_base = sbp_base - dbp_base
        pp_narrowing = 0.20 * map_drop_mag * (bp_drop_curve * is_post)
        pp_signal = np.maximum(22.0, pp_base - pp_narrowing + 0.6 * resp_mod)

        # SBP and DBP mathematically locked to MAP: MAP = DBP + PP/3
        sbp_signal = map_signal + (2.0 / 3.0) * pp_signal
        dbp_signal = map_signal - (1.0 / 3.0) * pp_signal

        # -------------------------------------------------------------
        # 3. Heart Rate Dynamics (Tachycardia / Baroreflex)
        # -------------------------------------------------------------
        if is_t2d:
            # Blunted baroreceptor response in T2D
            hr_peak_inc = float(self.rng.uniform(10.0, 25.0))
            t_peak_hr = float(self.rng.uniform(10.0, 18.0))
            hr_sustained = float(self.rng.uniform(6.0, 14.0))
            hr_noise_std = float(meta["baseline_rmssd"]) / 35.0  # Reduced HRV
        else:
            # Crisp, healthy baroreflex response in Controls
            hr_peak_inc = float(self.rng.uniform(16.0, 32.0))
            t_peak_hr = float(self.rng.uniform(7.0, 12.0))
            hr_sustained = float(self.rng.uniform(10.0, 18.0))
            hr_noise_std = float(meta["baseline_rmssd"]) / 20.0  # Higher HRV

        hr_surge_curve = (t_post / t_peak_hr) * np.exp(1.0 - (t_post / t_peak_hr))
        hr_steady_trans = 1.0 / (1.0 + np.exp(-(t - 12.0) / 4.0)) * is_post

        hr_ortho = (hr_peak_inc * hr_surge_curve * 0.7 + hr_sustained * hr_steady_trans * 0.3) * is_post
        rsa = (float(meta["baseline_rmssd"]) / 12.0) * resp_mod
        hr_signal = hr_base + hr_ortho + rsa + self.rng.normal(0, hr_noise_std, n_samples)
        hr_signal = np.clip(hr_signal, 40.0, 160.0)

        # -------------------------------------------------------------
        # 4. MCA Blood-Flow Velocity (MCAv, cm/s)
        # Dynamic Cerebral Autoregulation (dCA) Model
        # GROUND TRUTH FOR TRAINING ONLY
        # -------------------------------------------------------------
        if is_t2d:
            # Impaired dCA: larger drop, delayed nadir, delayed/incomplete recovery
            drop_pct = float(self.rng.uniform(self.t2d_drop_pct[0], self.t2d_drop_pct[1])) / 100.0
            t_nadir_mcav = float(self.rng.uniform(8.0, 15.0))
            tau_rec_mcav = float(self.rng.uniform(25.0, 50.0))
            autoreg_efficiency = float(self.rng.uniform(0.35, 0.65))  # Blunted vasodilation
            overshoot_amp = 0.0  # No rebound overshoot
        else:
            # Intact dCA: smaller drop, rapid nadir, rapid autoregulatory restoration
            drop_pct = float(self.rng.uniform(self.ctl_drop_pct[0], self.ctl_drop_pct[1])) / 100.0
            t_nadir_mcav = float(self.rng.uniform(5.0, 11.0))
            tau_rec_mcav = float(self.rng.uniform(9.0, 15.0))
            autoreg_efficiency = float(self.rng.uniform(0.85, 1.05))  # Strong arteriolar vasodilation
            overshoot_amp = float(self.rng.uniform(1.0, 3.5))  # Slight regulatory rebound

        # MCAv baseline velocity drop
        mcav_delta_max = mcav_base * drop_pct
        mcav_drop_curve = (t_post / t_nadir_mcav) * np.exp(1.0 - (t_post / t_nadir_mcav))

        # Dynamic autoregulatory recovery: cerebrovascular resistance (CVR) compensation
        # Restores flow velocity even if MAP remains slightly lower
        recovery_factor = (1.0 - np.exp(-t_post / max(tau_rec_mcav, 1e-3))) * autoreg_efficiency

        # Autoregulatory rebound overshoot for controls
        overshoot_curve = (
            overshoot_amp
            * np.sin(np.pi * np.clip((t_post - 12.0) / 25.0, 0.0, 1.0))
            * (t_post > 12.0).astype(float)
        )

        mcav_transient = -mcav_delta_max * mcav_drop_curve * (1.0 - 0.45 * recovery_factor)
        mcav_recovery = mcav_delta_max * recovery_factor * 0.45

        # Carbon dioxide reactivity coupling: EtCO2 decrease causes cerebral vasoconstriction
        co2_vaso_effect = 0.5 * (etco2_signal - etco2_base)

        # Pulse transmission from MAP
        bp_coupling = 0.12 if is_t2d else 0.05
        bp_transfer = bp_coupling * (map_signal - map_base)

        mcav_noise = self.rng.normal(0, 0.85, n_samples)
        mcav_signal = (
            mcav_base
            + (mcav_transient + mcav_recovery + overshoot_curve) * is_post
            + co2_vaso_effect
            + bp_transfer
            + 0.6 * resp_mod
            + mcav_noise
        )
        mcav_signal = np.clip(mcav_signal, 15.0, 120.0)

        # Assemble DataFrame with strict standardized naming
        waveform_df = pd.DataFrame(
            {
                "time_s": np.round(t, 4),
                "ecg_hr_bpm": np.round(hr_signal, 2),
                "sbp_mmhg": np.round(sbp_signal, 2),
                "dbp_mmhg": np.round(dbp_signal, 2),
                "map_mmhg": np.round(map_signal, 2),
                "resp_rate_bpm": np.round(rr_signal, 2),
                "etco2_mmhg": np.round(etco2_signal, 2),
                "mcav_cms": np.round(mcav_signal, 2),
            }
        )

        return waveform_df

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Generate metadata and waveforms for all 39 subjects.

        Returns:
            Tuple of:
            - subjects_metadata_df (pd.DataFrame): Demographics and autonomic baseline.
            - waveforms_dict (Dict[str, pd.DataFrame]): Mapping subject_id -> 6000-sample waveform DataFrame.
        """
        metadata_records: List[Dict[str, Any]] = []
        waveforms_dict: Dict[str, pd.DataFrame] = {}

        # S0001 - S0020: T2D (20 subjects)
        for i in range(1, self.n_t2d + 1):
            meta = self._sample_subject_metadata(subject_idx=i, group="T2D")
            metadata_records.append(meta)
            wf = self._generate_subject_waveforms(meta)
            waveforms_dict[meta["subject_id"]] = wf

        # S0021 - S0039: Controls (19 subjects)
        for i in range(self.n_t2d + 1, self.n_subjects + 1):
            meta = self._sample_subject_metadata(subject_idx=i, group="Control")
            metadata_records.append(meta)
            wf = self._generate_subject_waveforms(meta)
            waveforms_dict[meta["subject_id"]] = wf

        self.metadata_df = pd.DataFrame(metadata_records)
        self.waveforms = waveforms_dict

        return self.metadata_df, self.waveforms

    def save(self, output_dir: Optional[str] = None) -> None:
        """Save synthesized cohort metadata and per-subject waveform CSV files.

        Args:
            output_dir: Target directory path. If None, uses synthetic_dir from config.
        """
        if self.metadata_df is None or self.waveforms is None:
            self.generate()

        if output_dir is None:
            output_dir = self.config.get("data", {}).get("synthetic_dir", "data/synthetic")

        out_path = Path(output_dir)
        waveforms_dir = out_path / "waveforms"
        waveforms_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata.csv (converting list fields to JSON strings for CSV compatibility)
        export_meta = self.metadata_df.copy()
        export_meta["medications"] = export_meta["medications"].apply(lambda m: ";".join(m) if isinstance(m, list) else m)
        export_meta["comorbidities"] = export_meta["comorbidities"].apply(lambda c: ";".join(c) if isinstance(c, list) else c)

        meta_file = out_path / "metadata.csv"
        export_meta.to_csv(meta_file, index=False)

        # Save individual subject waveform CSV files
        for subject_id, wf_df in self.waveforms.items():
            wf_file = waveforms_dir / f"{subject_id}.csv"
            wf_df.to_csv(wf_file, index=False)

        print(f"[SyntheticCohortGenerator] Successfully generated and saved {len(self.metadata_df)} subjects to: {out_path.resolve()}")
