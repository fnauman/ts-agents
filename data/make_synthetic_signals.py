#!/usr/bin/env python3
"""
Generate a small known-answer signal table for non-forecasting examples.

Outputs a CSV with one row per sample at 50 Hz (t in seconds) and columns:

  t,periodic,regime,anomaly

Ground truth
------------
- periodic: 2 Hz sine (period 0.5 s = 25 samples) plus Gaussian noise (sd 0.3).
- regime:   white noise whose mean shifts by +2 at t = 20 s (sample 1000) and
            whose standard deviation triples at t = 40 s (sample 2000).
- anomaly:  0.5 Hz sine (period 100 samples) plus noise (sd 0.1), with the
            oscillation replaced by a flat line for samples 1500-1529.

Example
-------
python data/make_synthetic_signals.py --out data/synthetic_signals.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 50
N_SAMPLES = 3000


def make_signals(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / SAMPLE_RATE_HZ

    periodic = np.sin(2 * np.pi * 2.0 * t) + 0.3 * rng.standard_normal(N_SAMPLES)

    regime = rng.standard_normal(N_SAMPLES)
    regime[1000:] += 2.0
    regime[2000:] = 2.0 + 3.0 * (regime[2000:] - 2.0)

    anomaly = np.sin(2 * np.pi * 0.5 * t) + 0.1 * rng.standard_normal(N_SAMPLES)
    anomaly[1500:1530] = anomaly[1500] + 0.1 * rng.standard_normal(30)

    return pd.DataFrame(
        {
            "t": np.round(t, 4),
            "periodic": np.round(periodic, 5),
            "regime": np.round(regime, 5),
            "anomaly": np.round(anomaly, 5),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/synthetic_signals.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    make_signals(args.seed).to_csv(args.out, index=False)
    print(f"Wrote {N_SAMPLES} samples to {args.out}")


if __name__ == "__main__":
    main()
