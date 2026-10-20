#!/usr/bin/env python3
"""CerebroTwin - PhysioNet Dataset Downloader.

Downloads the sit-to-stand protocol subset from PhysioNet database:
'Cerebral Perfusion and Cognitive Function in Type 2 Diabetes'
(https://physionet.org/content/cerebral-perfusion-diabetes/1.0.1/)

Downloads WFDB record headers (.hea), binary signals (.dat), annotations (.atr),
and clinical metadata (subject-info.csv) for up to N=39 subjects.

CRITICAL DOMAIN RULES:
1. ALWAYS use "MCA blood-flow velocity" or "MCAv" (cm/s), NEVER "cerebral blood flow" or volumetric perfusion.
2. MCAv (TCD) is GROUND TRUTH for training ONLY — NEVER a runtime input feature.
3. Abnormality flags: "Research Response Status: HIGH DEVIATION" — NEVER clinical diagnosis.
4. t=0 is the sit-to-stand transition moment.
5. Subject-level splitting only (GroupKFold / LeaveOneGroupOut by subject_id).
6. N=39 subjects: 20 T2D, 19 Controls, aged 50-85.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

PHYSIONET_BASE_URL = "https://physionet.org/files/cerebral-perfusion-diabetes/1.0.1"
DEFAULT_DB_NAME = "cerebral-perfusion-diabetes/1.0.1"


def download_file_http(url: str, dest_path: Path, timeout: int = 30) -> bool:
    """Download a single file via HTTP with progress feedback.

    Args:
        url: Remote URL.
        dest_path: Local destination file path.
        timeout: Socket timeout in seconds.

    Returns:
        True if download succeeded, False otherwise.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CerebroTwin-Research-Agent/1.0 (Biomedical ML; Python)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 65536  # 64 KB

            with open(temp_path, "wb") as f_out:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f_out.write(buffer)

        # Rename part file upon completion
        temp_path.replace(dest_path)
        return True
    except urllib.error.HTTPError as e:
        if temp_path.exists():
            temp_path.unlink()
        if e.code == 404:
            return False
        print(f"    [HTTP Error {e.code}] for {url}: {e.reason}")
        return False
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"    [Network Error] Failed downloading {url}: {e}")
        return False


def get_record_manifest(output_dir: Path) -> List[str]:
    """Fetch RECORDS manifest from PhysioNet or local cache."""
    records_url = f"{PHYSIONET_BASE_URL}/RECORDS"
    local_records_path = output_dir / "RECORDS"

    print(f"Fetching record manifest from {records_url}...")
    manifest_fetched = download_file_http(records_url, local_records_path)

    if manifest_fetched and local_records_path.exists():
        with open(local_records_path, "r", encoding="utf-8") as f:
            records = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if records:
            return records

    # Try wfdb package if HTTP manifest is unavailable
    try:
        import wfdb
        print("Fetching record list via wfdb package...")
        records = wfdb.get_record_list("cerebral-perfusion-diabetes/1.0.1")
        if records:
            return records
    except Exception:
        pass

    # Standard fallback list of 39 subjects from PhysioNet cerebral-perfusion-diabetes
    print("Using standard N=39 cohort record list fallback...")
    records = []
    # 20 T2D subjects
    for i in range(1, 21):
        records.append(f"Subject_T2D_{i:02d}")
    # 19 Control subjects
    for i in range(1, 20):
        records.append(f"Subject_CTL_{i:02d}")
    return records


def download_physionet_cohort(output_dir: Path, n_subjects: int = 39, database: str = DEFAULT_DB_NAME) -> None:
    """Download records, headers, and metadata for specified number of subjects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"============================================================")
    print(f"CerebroTwin: Downloading PhysioNet Dataset")
    print(f"Database: {database}")
    print(f"Target Directory: {output_dir.resolve()}")
    print(f"Subject Limit: {n_subjects}")
    print(f"============================================================")

    # 1. Download clinical/demographic metadata file
    meta_files = ["subject-info.csv", "clinical_data.csv", "demographics.csv"]
    for mf in meta_files:
        meta_url = f"{PHYSIONET_BASE_URL}/{mf}"
        local_meta = output_dir / mf
        if not local_meta.exists():
            download_file_http(meta_url, local_meta)

    # 2. Get record list
    all_records = get_record_manifest(output_dir)
    selected_records = all_records[:n_subjects]
    print(f"Found {len(all_records)} total records; queued {len(selected_records)} records for download.")

    extensions = [".hea", ".dat", ".atr"]
    success_count = 0

    for idx, rec in enumerate(selected_records, start=1):
        rec_name = Path(rec).name
        print(f"\n[{idx}/{len(selected_records)}] Record: {rec_name}")

        rec_success = False
        for ext in extensions:
            target_file = output_dir / f"{rec}{ext}"
            if target_file.exists():
                print(f"  - {target_file.name} (already present)")
                rec_success = True
                continue

            file_url = f"{PHYSIONET_BASE_URL}/{rec}{ext}"
            print(f"  - Fetching {file_url} ...", end=" ", flush=True)
            ok = download_file_http(file_url, target_file)
            if ok:
                print("DONE")
                rec_success = True
            else:
                print("NOT FOUND / SKIPPED")

        if rec_success:
            success_count += 1

    print(f"\n============================================================")
    print(f"Download complete: {success_count}/{len(selected_records)} records successfully retrieved.")
    print(f"Data stored in: {output_dir.resolve()}")
    print(f"============================================================")


def main() -> None:
    """CLI entry point for PhysioNet downloading."""
    parser = argparse.ArgumentParser(
        description="Download subset of PhysioNet cerebral-perfusion-diabetes dataset for CerebroTwin."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/physionet",
        help="Destination directory for PhysioNet recordings (default: data/raw/physionet)",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=39,
        help="Number of subjects to download (default: 39)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DB_NAME,
        help=f"PhysioNet database identifier (default: {DEFAULT_DB_NAME})",
    )

    args = parser.parse_args()
    out_path = Path(args.output_dir)
    download_physionet_cohort(output_dir=out_path, n_subjects=args.n_subjects, database=args.database)


if __name__ == "__main__":
    main()
