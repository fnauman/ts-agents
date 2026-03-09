#!/usr/bin/env python3
"""
Generate synthetic labeled-stream time series for windowed classification demos.

Outputs a CSV with one row per timepoint and columns:
  timestamp,x,y,z,label

Why this exists
---------------
For window-size selection demos, you want:
- a *single long* multivariate time series (like sensor telemetry)
- a per-timepoint label stream
- variable segment lengths + class imbalance (so windowing is non-trivial)
- small enough to run fast in a CLI demo

Examples
--------
# "gait" (idle/walk/jog) at 20Hz for ~6 minutes
python data/make_synthetic_labeled_stream.py --scenario gait --minutes 6 --out data/demo_labeled_stream.csv

# "stairs" (stand/walk/up/down) at 25Hz
python data/make_synthetic_labeled_stream.py --scenario stairs --hz 25 --minutes 8 --out data/demo_labeled_stream.csv

# "industrial" (idle/vibration/impact) at 100Hz
python data/make_synthetic_labeled_stream.py --scenario industrial --hz 100 --seconds 90 --out data/demo_labeled_stream.csv
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SegmentSpec:
    label: str
    n: int


def _rand_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random 3D rotation matrix (uniform-ish) using random unit quaternion."""
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    # quaternion to rotation
    R = np.array([
        [1 - 2*(q3*q3 + q4*q4),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        [    2*(q2*q3 + q1*q4), 1 - 2*(q2*q2 + q4*q4),     2*(q3*q4 - q1*q2)],
        [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2*q2 + q3*q3)],
    ], dtype=float)
    return R


def _piecewise_segments(
    total_n: int,
    *,
    rng: np.random.Generator,
    class_probs: Dict[str, float],
    length_dists: Dict[str, Tuple[int, int]],
    min_segment_n: int = 20,
) -> List[SegmentSpec]:
    """Create a list of segments with variable lengths and class imbalance."""
    labels = list(class_probs.keys())
    probs = np.array([class_probs[k] for k in labels], dtype=float)
    probs = probs / probs.sum()

    segments: List[SegmentSpec] = []
    n_left = total_n
    while n_left > 0:
        lbl = rng.choice(labels, p=probs)
        lo, hi = length_dists[lbl]
        seg_n = int(rng.integers(lo, hi + 1))
        seg_n = max(seg_n, min_segment_n)
        seg_n = min(seg_n, n_left)
        segments.append(SegmentSpec(label=str(lbl), n=int(seg_n)))
        n_left -= seg_n
    return segments


def _gen_gait(segments: List[SegmentSpec], *, hz: float, rng: np.random.Generator) -> Tuple[np.ndarray, List[str]]:
    """idle/walk/jog IMU-like 3-axis series with per-segment random orientation."""
    X_parts: List[np.ndarray] = []
    y_parts: List[str] = []

    # Low-frequency drift shared across all segments.
    drift = np.zeros((sum(s.n for s in segments), 3), dtype=float)
    step = rng.normal(0, 0.005, size=drift.shape)
    drift = np.cumsum(step, axis=0)

    idx = 0
    for seg in segments:
        t = (np.arange(seg.n) / hz).astype(float)
        if seg.label == "idle":
            base = rng.normal(0, 0.05, size=(seg.n, 3))
        elif seg.label == "walk":
            f = rng.uniform(1.4, 2.2)  # ~steps/sec
            A = rng.uniform(0.8, 1.2)
            ph = rng.uniform(0, 2*math.pi, size=3)
            base = np.column_stack([
                A * np.sin(2*math.pi*f*t + ph[0]),
                0.8*A * np.sin(2*math.pi*f*t + ph[1] + 0.7),
                0.6*A * np.sin(2*math.pi*f*t + ph[2] + 1.3),
            ])
            base += 0.15 * A * np.sin(2*math.pi*(2*f)*t)[:, None]  # harmonic
            base += rng.normal(0, 0.08, size=base.shape)
        elif seg.label == "jog":
            f = rng.uniform(2.3, 3.3)
            A = rng.uniform(1.2, 1.8)
            ph = rng.uniform(0, 2*math.pi, size=3)
            base = np.column_stack([
                A * np.sin(2*math.pi*f*t + ph[0]),
                0.9*A * np.sin(2*math.pi*f*t + ph[1] + 0.4),
                0.7*A * np.sin(2*math.pi*f*t + ph[2] + 1.1),
            ])
            base += 0.25 * A * np.sin(2*math.pi*(3*f)*t)[:, None]
            base += rng.normal(0, 0.12, size=base.shape)
        else:
            raise ValueError(f"unknown gait label: {seg.label}")

        # Random orientation per segment (like phone in pocket vs hand).
        R = _rand_rotation(rng)
        rotated = base @ R.T

        # Add drift slice.
        rotated = rotated + drift[idx:idx+seg.n, :]

        X_parts.append(rotated)
        y_parts.extend([seg.label] * seg.n)
        idx += seg.n

    X = np.vstack(X_parts).astype(float)
    return X, y_parts


def _gen_stairs(segments: List[SegmentSpec], *, hz: float, rng: np.random.Generator) -> Tuple[np.ndarray, List[str]]:
    """stand/walk/stairs_up/stairs_down with asymmetric step patterns."""
    X_parts: List[np.ndarray] = []
    y_parts: List[str] = []

    def sawtooth(tt: np.ndarray, f: float) -> np.ndarray:
        # simple sawtooth in [-1,1]
        phase = (tt * f) % 1.0
        return 2 * phase - 1

    for seg in segments:
        t = (np.arange(seg.n) / hz).astype(float)
        if seg.label == "stand":
            base = rng.normal(0, 0.04, size=(seg.n, 3))
        elif seg.label == "walk":
            f = rng.uniform(1.5, 2.2)
            A = rng.uniform(0.7, 1.1)
            base = np.column_stack([
                A * np.sin(2*math.pi*f*t),
                0.7*A * np.sin(2*math.pi*f*t + 0.8),
                0.5*A * np.sin(2*math.pi*f*t + 1.5),
            ])
            base += rng.normal(0, 0.10, size=base.shape)
        elif seg.label == "stairs_up":
            f = rng.uniform(1.2, 1.8)
            A = rng.uniform(0.9, 1.4)
            s = sawtooth(t, f)
            base = np.column_stack([
                A * s,
                0.6*A * np.sin(2*math.pi*f*t + 0.4),
                0.8*A * np.maximum(0, np.sin(2*math.pi*f*t))  # upward bursts
            ])
            base += rng.normal(0, 0.12, size=base.shape)
        elif seg.label == "stairs_down":
            f = rng.uniform(1.2, 1.8)
            A = rng.uniform(0.9, 1.4)
            s = -sawtooth(t, f)
            base = np.column_stack([
                A * s,
                0.6*A * np.sin(2*math.pi*f*t + 1.0),
                0.8*A * np.minimum(0, np.sin(2*math.pi*f*t))  # downward bursts
            ])
            base += rng.normal(0, 0.12, size=base.shape)
        else:
            raise ValueError(f"unknown stairs label: {seg.label}")

        R = _rand_rotation(rng)
        X_parts.append(base @ R.T)
        y_parts.extend([seg.label] * seg.n)

    return np.vstack(X_parts).astype(float), y_parts


def _gen_industrial(segments: List[SegmentSpec], *, hz: float, rng: np.random.Generator) -> Tuple[np.ndarray, List[str]]:
    """idle / vibration / impact with bursts + harmonics (3-axis)."""
    X_parts: List[np.ndarray] = []
    y_parts: List[str] = []

    for seg in segments:
        t = (np.arange(seg.n) / hz).astype(float)

        if seg.label == "idle":
            base = rng.normal(0, 0.02, size=(seg.n, 3))
        elif seg.label == "vibration":
            f = rng.uniform(12, 25)  # Hz
            A = rng.uniform(0.4, 0.8)
            base = np.column_stack([
                A * np.sin(2*math.pi*f*t),
                0.7*A * np.sin(2*math.pi*(f*1.1)*t + 0.3),
                0.6*A * np.sin(2*math.pi*(f*0.9)*t + 1.1),
            ])
            base += 0.25 * A * np.sin(2*math.pi*(2*f)*t)[:, None]
            base += rng.normal(0, 0.05, size=base.shape)
        elif seg.label == "impact":
            base = rng.normal(0, 0.03, size=(seg.n, 3))
            # Add several spikes (impacts) with decays
            n_spikes = int(rng.integers(2, 6))
            for _ in range(n_spikes):
                center = int(rng.integers(0, seg.n))
                width = float(rng.uniform(0.01, 0.05))  # seconds
                sigma = max(1, int(width * hz))
                amp = float(rng.uniform(1.0, 2.5))
                spike = amp * np.exp(-0.5 * ((np.arange(seg.n) - center) / sigma) ** 2)
                axis = int(rng.integers(0, 3))
                base[:, axis] += spike
            base += rng.normal(0, 0.06, size=base.shape)
        else:
            raise ValueError(f"unknown industrial label: {seg.label}")

        R = _rand_rotation(rng)
        X_parts.append(base @ R.T)
        y_parts.extend([seg.label] * seg.n)

    return np.vstack(X_parts).astype(float), y_parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["gait", "stairs", "industrial"], default="gait")
    ap.add_argument("--hz", type=float, default=20.0, help="Sampling rate (Hz)")
    ap.add_argument("--minutes", type=float, default=None, help="Total length in minutes (preferred)")
    ap.add_argument("--seconds", type=float, default=None, help="Total length in seconds (overrides minutes)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, required=True)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.seconds is not None:
        total_seconds = float(args.seconds)
    elif args.minutes is not None:
        total_seconds = float(args.minutes) * 60.0
    else:
        total_seconds = 6.0 * 60.0

    total_n = int(max(1, round(total_seconds * float(args.hz))))

    if args.scenario == "gait":
        class_probs = {"idle": 0.60, "walk": 0.30, "jog": 0.10}
        length_dists = {
            "idle": (int(4*args.hz), int(12*args.hz)),
            "walk": (int(3*args.hz), int(9*args.hz)),
            "jog": (int(1*args.hz), int(4*args.hz)),
        }
        segments = _piecewise_segments(total_n, rng=rng, class_probs=class_probs, length_dists=length_dists)
        X, y = _gen_gait(segments, hz=float(args.hz), rng=rng)
    elif args.scenario == "stairs":
        class_probs = {"stand": 0.45, "walk": 0.25, "stairs_up": 0.15, "stairs_down": 0.15}
        length_dists = {
            "stand": (int(5*args.hz), int(15*args.hz)),
            "walk": (int(3*args.hz), int(10*args.hz)),
            "stairs_up": (int(2*args.hz), int(6*args.hz)),
            "stairs_down": (int(2*args.hz), int(6*args.hz)),
        }
        segments = _piecewise_segments(total_n, rng=rng, class_probs=class_probs, length_dists=length_dists)
        X, y = _gen_stairs(segments, hz=float(args.hz), rng=rng)
    else:
        class_probs = {"idle": 0.55, "vibration": 0.35, "impact": 0.10}
        length_dists = {
            "idle": (int(1.0*args.hz), int(4.0*args.hz)),
            "vibration": (int(1.0*args.hz), int(4.0*args.hz)),
            "impact": (int(0.5*args.hz), int(2.0*args.hz)),
        }
        segments = _piecewise_segments(total_n, rng=rng, class_probs=class_probs, length_dists=length_dists)
        X, y = _gen_industrial(segments, hz=float(args.hz), rng=rng)

    ts = np.arange(total_n, dtype=np.int64)
    df = pd.DataFrame({
        "timestamp": ts,
        "x": X[:, 0],
        "y": X[:, 1],
        "z": X[:, 2],
        "label": y,
    })

    out_path = args.out
    pd.options.mode.chained_assignment = None
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print("Label counts:", df["label"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
