#!/usr/bin/env python3
"""
Create a multi-subject WISDM labeled stream CSV from the raw dataset.

Builds on the same raw data as make_demo_labeled_stream_wisdm.py but concatenates
data from many subjects for a richer activity-recognition dataset.

Examples
--------
# Default: 15 subjects, 6 activities (walking/jogging/stairs/sitting/standing/clapping), phone accel
python data/make_wisdm_labeled_stream.py

# Custom subject count and output path
python data/make_wisdm_labeled_stream.py --n-subjects 20 --out data/wisdm_large.csv

# All 48 eligible subjects
python data/make_wisdm_labeled_stream.py --n-subjects all

# Small ~2 MB subset: 2 subjects, trim each activity to 140s
python data/make_wisdm_labeled_stream.py --n-subjects 2 --seconds-per-activity 140 --out data/wisdm_subset.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Reuse helpers from the demo script
from make_demo_labeled_stream_wisdm import (
    _find_activity_key,
    _parse_activity_key,
    _read_wisdm_file,
    _wisdm_data_path,
)

DEFAULT_ACTIVITIES = ["walking", "jogging", "stairs", "sitting", "standing", "clapping"]
ACTIVITY_CODE_TO_NAME = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}


def _find_eligible_subjects(
    root: Path,
    *,
    device: str,
    sensor: str,
    activity_codes: set[str],
) -> List[int]:
    """Return subject IDs that have all requested activity codes."""
    data_dir = root / "wisdm-dataset" / "raw" / device / sensor
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    eligible = []
    for f in sorted(data_dir.glob("data_*_*.txt")):
        subj = int(f.name.split("_")[1])
        codes_found: set[str] = set()
        with open(f) as fh:
            for line in fh:
                parts = line.strip().rstrip(";").split(",")
                if len(parts) >= 2:
                    code = parts[1].strip()
                    if code in activity_codes:
                        codes_found.add(code)
                    if codes_found == activity_codes:
                        break
        if activity_codes.issubset(codes_found):
            eligible.append(subj)
    return eligible


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a multi-subject WISDM labeled stream CSV."
    )
    ap.add_argument(
        "--dataset-dir",
        type=str,
        default="data/raw/wisdm",
        help="Root directory of the extracted WISDM dataset",
    )
    ap.add_argument("--device", choices=["phone", "watch"], default="phone")
    ap.add_argument("--sensor", choices=["accel", "gyro"], default="accel")
    ap.add_argument(
        "--activities",
        type=str,
        default=",".join(DEFAULT_ACTIVITIES),
        help="Comma-separated activity names to include",
    )
    ap.add_argument(
        "--n-subjects",
        type=str,
        default="15",
        help="Number of subjects to include, or 'all'",
    )
    ap.add_argument(
        "--seconds-per-activity",
        type=float,
        default=None,
        help="Trim each activity to this many seconds per subject (at 20 Hz). "
        "Keeps contiguous blocks from the start of each activity segment.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data/wisdm_labeled_stream.csv",
    )
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    if not root.exists():
        print(f"ERROR: Dataset directory not found: {root}", file=sys.stderr)
        print("Run make_demo_labeled_stream_wisdm.py first to download/extract.", file=sys.stderr)
        return 1

    # Parse activity key
    key_path = _find_activity_key(root)
    code_to_name = _parse_activity_key(key_path)

    # Build reverse map (name -> code) for filtering
    name_to_code: Dict[str, str] = {v.lower(): k for k, v in code_to_name.items()}
    requested_names = [a.strip().lower() for a in args.activities.split(",") if a.strip()]
    requested_codes = set()
    for name in requested_names:
        if name in name_to_code:
            requested_codes.add(name_to_code[name])
        else:
            # Try direct code lookup
            upper = name.upper()
            if upper in code_to_name:
                requested_codes.add(upper)
            else:
                print(f"WARNING: Unknown activity '{name}', skipping", file=sys.stderr)

    if not requested_codes:
        print("ERROR: No valid activities specified.", file=sys.stderr)
        return 1

    print(f"Target activities: {', '.join(code_to_name[c] for c in sorted(requested_codes))}")

    # Find eligible subjects
    print(f"Scanning {args.device}/{args.sensor} for eligible subjects ...")
    eligible = _find_eligible_subjects(
        root, device=args.device, sensor=args.sensor, activity_codes=requested_codes
    )
    print(f"Found {len(eligible)} subjects with all {len(requested_codes)} activities")

    if not eligible:
        print("ERROR: No eligible subjects found.", file=sys.stderr)
        return 1

    # Select subjects
    if args.n_subjects.lower() == "all":
        subjects = eligible
    else:
        n = int(args.n_subjects)
        if n <= 0:
            print("ERROR: --n-subjects must be positive.", file=sys.stderr)
            return 1
        subjects = eligible[:n]

    print(f"Using {len(subjects)} subjects: {subjects[0]}..{subjects[-1]}")

    # Load and concatenate data
    frames: List[pd.DataFrame] = []
    for subj in subjects:
        data_path = _wisdm_data_path(root, subject=subj, device=args.device, sensor=args.sensor)
        df = _read_wisdm_file(data_path)
        df["label"] = df["activity_code"].map(code_to_name).fillna(df["activity_code"].astype(str))
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        # Filter to requested activities
        df = df[df["activity_code"].isin(requested_codes)].copy()
        frames.append(df)
        print(f"  Subject {subj}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows: {len(combined):,}")

    # Optional per-activity trimming
    if args.seconds_per_activity is not None:
        nominal_hz = 20.0
        max_rows = int(args.seconds_per_activity * nominal_hz)
        trimmed_parts = []
        for subj in subjects:
            sdf = combined[combined["subject_id"] == subj]
            for label in sorted(sdf["label"].unique()):
                trimmed_parts.append(sdf[sdf["label"] == label].iloc[:max_rows])
        combined = pd.concat(trimmed_parts, ignore_index=True)
        print(f"After trimming to {args.seconds_per_activity}s per activity: {len(combined):,} rows")

    # Output columns: subject_id, timestamp, x, y, z, label
    out_df = combined[["subject_id", "timestamp", "x", "y", "z", "label"]].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nWrote {len(out_df):,} rows to {out_path}")
    print("\nLabel distribution:")
    vc = out_df["label"].value_counts()
    for lbl, cnt in vc.items():
        pct = 100 * cnt / len(out_df)
        print(f"  {lbl:12s}: {cnt:>8,} ({pct:5.1f}%)")

    print(f"\nSubjects: {len(subjects)}")
    print(f"Activities: {len(requested_codes)}")
    print(f"Sample rate: 20 Hz")
    print(f"Duration per subject per activity: ~3 min")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
