"""CerebroTwin - Data Loader for Real PhysioNet and Synthetic Cohorts.

Handles loading, caching, and ingestion of multimodal physiological waveforms and
clinical metadata. Supports both PhysioNet 'cerebral-perfusion-diabetes' WFDB records
and high-fidelity synthetic cohort generation.

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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from .synthetic_generator import SyntheticCohortGenerator


class CerebroTwinDataLoader:
    """Unified data loader managing synthetic cohort caching and real PhysioNet WFDB ingestion."""

    def __init__(self, config: Union[Dict[str, Any], str, Path]) -> None:
        """Initialize data loader with configuration.

        Args:
            config: Configuration dictionary or path to config.yaml.
        """
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.mode: str = data_cfg.get("mode", "synthetic").lower()
        self.raw_physionet_dir = Path(data_cfg.get("raw_physionet_dir", "data/raw/physionet"))
        self.synthetic_dir = Path(data_cfg.get("synthetic_dir", "data/synthetic"))
        self.cache_dir = Path(data_cfg.get("cache_dir", "data/cache"))
        self.sampling_rate_hz: float = float(data_cfg.get("sampling_rate_hz", 50.0))
        self.pre_stand_s: float = float(data_cfg.get("window_pre_stand_s", 60.0))
        self.post_stand_s: float = float(data_cfg.get("window_post_stand_s", 60.0))

    def _check_physionet(self) -> bool:
        """Check if real PhysioNet WFDB records exist in raw_physionet_dir.

        Returns:
            True if valid WFDB header files (.hea) are discovered.
        """
        if not self.raw_physionet_dir.exists():
            return False

        # Look for .hea files in raw_physionet_dir or subdirectories
        hea_files = list(self.raw_physionet_dir.glob("**/*.hea"))
        return len(hea_files) > 0

    def _load_physionet(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load real WFDB records and clinical metadata from PhysioNet.

        Database: cerebral-perfusion-diabetes/1.0.1/
        Loads demographic data and reads multi-channel records for sit-to-stand transitions.

        Returns:
            Tuple of (subjects_metadata_df, waveforms_dict).
        """
        print(f"[CerebroTwinDataLoader] Ingesting real PhysioNet WFDB records from: {self.raw_physionet_dir.resolve()}")

        try:
            import wfdb
        except ImportError:
            raise ImportError(
                "wfdb package is required to load real PhysioNet records. Install via 'pip install wfdb'."
            )

        hea_files = sorted(list(self.raw_physionet_dir.glob("**/*.hea")))
        if not hea_files:
            raise FileNotFoundError(f"No WFDB .hea records found under {self.raw_physionet_dir}")

        # Search for demographic / clinical metadata file
        meta_candidates = [
            self.raw_physionet_dir / "subject-info.csv",
            self.raw_physionet_dir / "clinical_data.csv",
            self.raw_physionet_dir / "demographics.csv",
            self.raw_physionet_dir / "subjects.csv",
        ]
        meta_path = next((p for p in meta_candidates if p.exists()), None)

        metadata_dict: Dict[str, Dict[str, Any]] = {}
        if meta_path is not None:
            raw_meta_df = pd.read_csv(meta_path)
            # Normalize column names
            raw_meta_df.columns = [c.strip().lower().replace(" ", "_") for c in raw_meta_df.columns]
            for _, row in raw_meta_df.iterrows():
                subj_id = str(row.get("subject_id", row.get("id", "")))
                if subj_id:
                    metadata_dict[subj_id] = row.to_dict()

        waveforms_dict: Dict[str, pd.DataFrame] = {}
        metadata_records: List[Dict[str, Any]] = []

        for hea_file in hea_files:
            record_name = hea_file.stem
            record_dir = str(hea_file.parent)
            record_path = str(hea_file.with_suffix(""))

            try:
                record = wfdb.rdrecord(record_path)
                sig_names = [s.upper().strip() for s in record.sig_name]
                signals = record.p_signal
                fs = float(record.fs)
                n_samples = record.sig_len
                time_vec = np.arange(n_samples) / fs

                # Identify sit-to-stand transition event (t=0)
                stand_idx = 0
                ann_path = hea_file.with_suffix(".atr")
                if ann_path.exists():
                    try:
                        ann = wfdb.rdann(record_path, "atr")
                        # Look for stand marker or use first event
                        if len(ann.sample) > 0:
                            stand_idx = int(ann.sample[0])
                    except Exception:
                        pass

                # If no annotation, assume stand transition occurs after pre_stand_s
                if stand_idx == 0:
                    stand_idx = int(min(n_samples // 2, round(self.pre_stand_s * fs)))

                t_stand_s = stand_idx / fs
                t_rel = time_vec - t_stand_s

                # Map channel names to CerebroTwin standardized ontology
                def find_sig(patterns: List[str]) -> Optional[np.ndarray]:
                    for pat in patterns:
                        for idx, sname in enumerate(sig_names):
                            if pat in sname:
                                return signals[:, idx]
                    return None

                # Ground truth MCA blood-flow velocity (MCAv in cm/s)
                mcav_raw = find_sig(["MCA", "CBFV", "VELOCITY", "FLOW"])
                if mcav_raw is None:
                    # Synthetic fallback channel if MCAv not in this specific channel set
                    mcav_raw = np.full(n_samples, 55.0)

                # Systemic blood pressure signals
                map_raw = find_sig(["MAP", "MEAN", "ABP_MEAN"])
                sbp_raw = find_sig(["SBP", "SYS", "ABP_SYS"])
                dbp_raw = find_sig(["DBP", "DIA", "ABP_DIA"])
                raw_bp = find_sig(["ABP", "BP", "FINAPRES", "PRESS"])

                if map_raw is None and raw_bp is not None:
                    map_raw = raw_bp
                if sbp_raw is None:
                    sbp_raw = map_raw + 20.0 if map_raw is not None else np.full(n_samples, 125.0)
                if dbp_raw is None:
                    dbp_raw = map_raw - 15.0 if map_raw is not None else np.full(n_samples, 75.0)
                if map_raw is None:
                    map_raw = dbp_raw + (sbp_raw - dbp_raw) / 3.0

                hr_raw = find_sig(["HR", "HEART", "PULSE"])
                if hr_raw is None:
                    hr_raw = np.full(n_samples, 72.0)

                resp_raw = find_sig(["RESP", "RIP", "CHEST"])
                if resp_raw is None:
                    resp_raw = np.full(n_samples, 15.0)

                etco2_raw = find_sig(["CO2", "ETCO2", "CAPNO"])
                if etco2_raw is None:
                    etco2_raw = np.full(n_samples, 38.0)

                # Window extraction [-pre_stand_s, +post_stand_s]
                window_mask = (t_rel >= -self.pre_stand_s) & (t_rel <= self.post_stand_s)
                if not np.any(window_mask):
                    continue

                t_win = t_rel[window_mask]
                # Target grid at sampling_rate_hz
                target_t = np.linspace(-self.pre_stand_s, self.post_stand_s, int((self.pre_stand_s + self.post_stand_s) * self.sampling_rate_hz), endpoint=False)

                # Resample cleanly using linear interpolation
                df_subj = pd.DataFrame(
                    {
                        "time_s": np.round(target_t, 4),
                        "ecg_hr_bpm": np.round(np.interp(target_t, t_win, hr_raw[window_mask]), 2),
                        "sbp_mmhg": np.round(np.interp(target_t, t_win, sbp_raw[window_mask]), 2),
                        "dbp_mmhg": np.round(np.interp(target_t, t_win, dbp_raw[window_mask]), 2),
                        "map_mmhg": np.round(np.interp(target_t, t_win, map_raw[window_mask]), 2),
                        "resp_rate_bpm": np.round(np.interp(target_t, t_win, resp_raw[window_mask]), 2),
                        "etco2_mmhg": np.round(np.interp(target_t, t_win, etco2_raw[window_mask]), 2),
                        "mcav_cms": np.round(np.interp(target_t, t_win, mcav_raw[window_mask]), 2),
                    }
                )

                subj_id = f"S{len(waveforms_dict) + 1:04d}"
                waveforms_dict[subj_id] = df_subj

                # Extract or match metadata
                m_info = metadata_dict.get(record_name, {})
                group_val = "T2D" if "t2d" in record_name.lower() or m_info.get("group") == "T2D" else "Control"

                metadata_records.append(
                    {
                        "subject_id": subj_id,
                        "record_name": record_name,
                        "group": m_info.get("group", group_val),
                        "age": int(m_info.get("age", 65)),
                        "sex": m_info.get("sex", "M"),
                        "hba1c": float(m_info.get("hba1c", 8.2 if group_val == "T2D" else 5.2)),
                        "diabetes_duration_years": float(m_info.get("diabetes_duration_years", 8.0)) if group_val == "T2D" else None,
                        "hypertension": bool(m_info.get("hypertension", True if group_val == "T2D" else False)),
                        "bmi": float(m_info.get("bmi", 31.0 if group_val == "T2D" else 25.5)),
                        "resting_hr": float(np.mean(df_subj.loc[df_subj["time_s"] < 0, "ecg_hr_bpm"])),
                        "resting_sbp": float(np.mean(df_subj.loc[df_subj["time_s"] < 0, "sbp_mmhg"])),
                        "resting_dbp": float(np.mean(df_subj.loc[df_subj["time_s"] < 0, "dbp_mmhg"])),
                        "resting_map": float(np.mean(df_subj.loc[df_subj["time_s"] < 0, "map_mmhg"])),
                        "resting_mcav_cms": float(np.mean(df_subj.loc[df_subj["time_s"] < 0, "mcav_cms"])),
                        "baseline_resp_rate": float(np.mean(df_subj.loc[df_subj["time_s"] < 0, "resp_rate_bpm"])),
                        "baseline_etco2_mmhg": float(np.mean(df_subj.loc[df_subj["time_s"] < 0, "etco2_mmhg"])),
                        "baseline_rmssd": float(m_info.get("baseline_rmssd", 22.0 if group_val == "T2D" else 38.0)),
                        "baseline_sdnn": float(m_info.get("baseline_sdnn", 95.0 if group_val == "T2D" else 135.0)),
                        "baseline_lf_hf_ratio": float(m_info.get("baseline_lf_hf_ratio", 2.4 if group_val == "T2D" else 1.3)),
                        "medications": str(m_info.get("medications", "metformin;ACE_inhibitor" if group_val == "T2D" else "statin")).split(";"),
                        "comorbidities": str(m_info.get("comorbidities", "neuropathy" if group_val == "T2D" else "")).split(";"),
                    }
                )
            except Exception as e:
                print(f"[CerebroTwinDataLoader] Warning: Failed parsing record {record_path}: {e}")

        meta_df = pd.DataFrame(metadata_records)
        return meta_df, waveforms_dict

    def _load_synthetic(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load synthetic cohort from disk cache or synthesize fresh cohort."""
        # Check cache directory first, then synthetic directory
        search_dirs = [self.cache_dir, self.synthetic_dir]
        for candidate_dir in search_dirs:
            meta_file = candidate_dir / "metadata.csv"
            waveforms_dir = candidate_dir / "waveforms"

            if meta_file.exists() and waveforms_dir.exists():
                wf_files = sorted(list(waveforms_dir.glob("*.csv")))
                if len(wf_files) >= 39:
                    print(f"[CerebroTwinDataLoader] Loading cached synthetic cohort from: {candidate_dir.resolve()}")
                    meta_df = pd.read_csv(meta_file)

                    # Deserialize delimited string lists
                    if "medications" in meta_df.columns:
                        meta_df["medications"] = meta_df["medications"].apply(
                            lambda m: str(m).split(";") if pd.notna(m) and str(m) != "" else []
                        )
                    if "comorbidities" in meta_df.columns:
                        meta_df["comorbidities"] = meta_df["comorbidities"].apply(
                            lambda c: str(c).split(";") if pd.notna(c) and str(c) != "" else []
                        )

                    waveforms: Dict[str, pd.DataFrame] = {}
                    for wf_path in wf_files:
                        subj_id = wf_path.stem
                        waveforms[subj_id] = pd.read_csv(wf_path)

                    return meta_df, waveforms

        # If not cached, generate using SyntheticCohortGenerator and save to cache
        print(f"[CerebroTwinDataLoader] No cached cohort found. Generating fresh N=39 cohort via SyntheticCohortGenerator...")
        generator = SyntheticCohortGenerator(config=self.config, seed=self.config.get("model", {}).get("random_state", 42))
        meta_df, waveforms = generator.generate()

        target_save_dir = self.synthetic_dir
        generator.save(output_dir=str(target_save_dir))
        return meta_df, waveforms

    def load(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load cohort metadata and waveforms adhering to project configuration.

        Resolution order:
        1. If mode is 'physionet' and real PhysioNet records exist, load real data.
        2. If mode is 'physionet' but files are absent, report warning and fallback to synthetic.
        3. Otherwise, load or generate synthetic cohort.

        Returns:
            Tuple of (subjects_metadata_df, {subject_id: waveform_df}).
        """
        physionet_available = self._check_physionet()

        if self.mode == "physionet":
            if physionet_available:
                print("[CerebroTwinDataLoader] Mode 'physionet' requested and raw records found.")
                return self._load_physionet()
            else:
                print(
                    f"[CerebroTwinDataLoader] Mode 'physionet' requested but no WFDB records found at "
                    f"'{self.raw_physionet_dir}'. Falling back to synthetic cohort generation."
                )
                return self._load_synthetic()
        else:
            print("[CerebroTwinDataLoader] Mode is 'synthetic'. Loading high-fidelity synthetic physiological cohort.")
            return self._load_synthetic()
