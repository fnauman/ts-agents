#!/usr/bin/env python3
"""
Create a small `demo_labeled_stream.csv` from the WISDM Smartphone+Smartwatch dataset (UCI).

Dataset
-------
- Name: WISDM Smartphone and Smartwatch Activity and Biometrics Dataset
- Raw format: <subject-id, activity-code, timestamp, x, y, z> (per sensor stream)
- License: CC BY 4.0 (see UCI page)

This script:
1) downloads the dataset zip (if needed)
2) extracts a single subject + sensor file
3) maps activity codes -> names using activity_key.txt
4) optionally trims segments to create *uneven* class durations (useful for window-size selection demos)
5) writes a compact CSV that ts-agents windowing tools can ingest.

Examples
--------
# Create a ~6-10 minute demo stream from watch accelerometer for subject 1600
python data/make_demo_labeled_stream_wisdm.py \
  --subject 1600 --device watch --sensor accel \
  --activities walking,jogging,sitting,standing \
  --trim-policy per_class_seconds \
  --per-class-seconds walking=180,jogging=60,sitting=180,standing=180 \
  --out data/demo_labeled_stream.csv

# Use phone accelerometer, keep original sequence and full segments (each is ~3 minutes)
python data/make_demo_labeled_stream_wisdm.py \
  --subject 1600 --device phone --sensor accel \
  --trim-policy as_is \
  --out data/demo_labeled_stream.csv

Notes
-----
- The raw WISDM files often have a trailing ';' at the end of lines. We strip it.
- The download is large (~295MB). For demos, you only keep one subject file.

"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import pandas as pd


DEFAULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"


def _parse_activity_key(path: Path) -> Dict[str, str]:
    """
    Parse activity_key.txt that maps codes to activity names.

    WISDM activity keys have appeared in at least two formats:

    - code-first:
        A = Walking
        B = Jogging
    - name-first:
        walking = A
        jogging = B

    We normalize to a mapping of *activity code* -> *activity name*.
    """
    mapping: Dict[str, str] = {}
    text = path.read_text(errors="ignore").splitlines()
    for line in text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comments
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

        # Try code-first: "A = Walking"
        m = re.match(r"^([A-Z])\s*=\s*(.+)$", line)
        if m:
            code = m.group(1).strip().upper()
            name = m.group(2).strip()
            if name:
                mapping[code] = name
            continue

        # Try name-first: "walking = A"
        m = re.match(r"^(.+?)\s*=\s*([A-Z])$", line)
        if m:
            name = m.group(1).strip()
            code = m.group(2).strip().upper()
            if name:
                mapping[code] = name
            continue

        # Try delimited variants, e.g. "A,Walking" or "walking,A"
        m = re.match(r"^([A-Z])\s*[,;]\s*(.+)$", line)
        if m:
            code = m.group(1).strip().upper()
            name = m.group(2).strip()
            if name:
                mapping[code] = name
            continue
        m = re.match(r"^(.+?)\s*[,;]\s*([A-Z])$", line)
        if m:
            name = m.group(1).strip()
            code = m.group(2).strip().upper()
            if name:
                mapping[code] = name
    if not mapping:
        raise ValueError(f"Failed to parse activity key from {path}")
    return mapping


def _find_activity_key(root: Path) -> Path:
    candidates = [
        root / "wisdm-dataset" / "activity_key.txt",
        root / "activity_key.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback search
    hits = list(root.rglob("activity_key.txt"))
    if hits:
        return hits[0]
    raise FileNotFoundError("activity_key.txt not found after extraction")


def _wisdm_data_path(root: Path, *, subject: int, device: str, sensor: str) -> Path:
    # roerich example: wisdm-dataset/raw/watch/accel/data_1600_accel_watch.txt
    rel = Path("wisdm-dataset") / "raw" / device / sensor / f"data_{subject}_{sensor}_{device}.txt"
    return root / rel


def _read_wisdm_file(path: Path) -> pd.DataFrame:
    cols = ["subject_id", "activity_code", "timestamp", "x", "y", "z"]
    # Read as strings first to safely strip trailing ';'
    df = pd.read_csv(
        path,
        names=cols,
        header=None,
        dtype=str,
        engine="python",
    )
    # Strip trailing semicolons/spaces
    for c in cols:
        df[c] = df[c].astype(str).str.strip().str.replace(";", "", regex=False)
    # Convert numeric columns
    df["subject_id"] = df["subject_id"].astype(int)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    for c in ["x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "x", "y", "z", "activity_code"])
    return df


def _compute_segments(labels: List[str]) -> List[Tuple[int, int, str]]:
    """Return segments as (start_idx, end_idx_exclusive, label)."""
    if not labels:
        return []
    segs: List[Tuple[int, int, str]] = []
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            segs.append((start, i, cur))
            start = i
            cur = labels[i]
    segs.append((start, len(labels), cur))
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-zip", type=str, default="data/raw/wisdm-dataset.zip", help="Path to store the dataset zip")
    ap.add_argument("--dataset-dir", type=str, default="data/raw/wisdm", help="Directory to extract into")
    ap.add_argument("--url", type=str, default=DEFAULT_URL, help="Download URL")
    ap.add_argument("--subject", type=int, default=1600)
    ap.add_argument("--device", choices=["phone", "watch"], default="watch")
    ap.add_argument("--sensor", choices=["accel", "gyro"], default="accel")
    ap.add_argument("--activities", type=str, default=None, help="Comma-separated activity *names* to keep (after code->name mapping)")
    ap.add_argument("--trim-policy", choices=["as_is", "per_class_seconds"], default="per_class_seconds")
    ap.add_argument("--per-class-seconds", type=str, default=None, help="Comma-separated name=seconds pairs (e.g. walking=180,jogging=60)")
    ap.add_argument("--out", type=str, default="data/demo_labeled_stream.csv")

    args = ap.parse_args()

    zip_path = Path(args.dataset_zip)
    extract_root = Path(args.dataset_dir)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading WISDM dataset to {zip_path} ...")
        urlretrieve(args.url, zip_path)
        print("Download complete.")

    # Extract if needed
    marker = extract_root / "wisdm-dataset"
    if not marker.exists():
        print(f"Extracting {zip_path} -> {extract_root} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
        print("Extraction complete.")

    key_path = _find_activity_key(extract_root)
    code_to_name = _parse_activity_key(key_path)

    data_path = _wisdm_data_path(extract_root, subject=args.subject, device=args.device, sensor=args.sensor)
    if not data_path.exists():
        raise FileNotFoundError(f"Expected WISDM file not found: {data_path}")

    df = _read_wisdm_file(data_path)
    # Map codes -> names (lowercase normalized)
    df["label"] = df["activity_code"].map(code_to_name).fillna(df["activity_code"].astype(str))
    df["label"] = df["label"].astype(str).str.strip()

    # Filter activities if requested
    keep_names: Optional[List[str]] = None
    if args.activities:
        keep_names = [a.strip() for a in args.activities.split(",") if a.strip()]
        # allow case-insensitive matching
        keep_lower = {k.lower(): k for k in keep_names}
        df = df[df["label"].str.lower().isin(set(keep_lower.keys()))].copy()
        # restore original casing from requested list (nice for demo)
        df["label"] = df["label"].str.lower().map(keep_lower)

    if df.empty:
        raise ValueError("No rows left after filtering. Check --activities or dataset availability.")

    # Optional trimming to create uneven segment durations (better demo for window selection)
    if args.trim_policy == "per_class_seconds":
        if not args.per_class_seconds:
            # sensible defaults if user didn't provide: keep 3 minutes for most, 1 minute for the "rare" class
            args.per_class_seconds = "walking=180,jogging=60,sitting=180,standing=180"
        per: Dict[str, int] = {}
        for pair in args.per_class_seconds.split(","):
            pair = pair.strip()
            if not pair:
                continue
            name, val = pair.split("=", 1)
            per[name.strip().lower()] = int(float(val.strip()))

        # Estimate sample rate from timestamps (WISDM raw is ~20Hz but timestamps can be large ints)
        # We'll trim by *row count* using a nominal hz=20 to avoid dealing with timestamp units.
        nominal_hz = 20.0

        labels = df["label"].tolist()
        segs = _compute_segments(labels)

        keep_idx: List[int] = []
        for start, end, lbl in segs:
            sec = per.get(lbl.lower())
            if sec is None:
                # If label not specified, keep full segment
                keep_idx.extend(range(start, end))
                continue
            max_rows = int(sec * nominal_hz)
            keep_idx.extend(range(start, min(end, start + max_rows)))

        df = df.iloc[keep_idx].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = df[["timestamp", "x", "y", "z", "label"]].copy()
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(out_df):,} rows to {out_path}")
    print("Labels:", out_df["label"].value_counts().to_dict())
    print()
    print("Next: run the activity-recognition workflow, e.g.")
    print("  uv run ts-agents workflow run activity-recognition \\")
    print(f"    --input {out_path} --label-col label --value-cols x,y,z \\")
    print("    --window-sizes 32,64,96,128,160 --classifier minirocket --metric balanced_accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
