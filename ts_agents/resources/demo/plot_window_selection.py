#!/usr/bin/env python3
"""
Plot window-size selection scores (ts-agents select_window_size[_from_csv] output).

Usage:
  python demo/plot_window_selection.py outputs/demo/window_selection.json outputs/demo/window_scores.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("selection_json", type=str)
    ap.add_argument("out_png", type=str)
    args = ap.parse_args()

    payload: Dict[str, Any] = json.loads(Path(args.selection_json).read_text())
    payload = payload.get("result", payload)
    scores = payload.get("scores_by_window") or {}
    if not scores:
        raise ValueError("No scores_by_window found in selection JSON.")

    # keys might be strings (JSON) -> cast to int
    items = sorted(((int(k), float(v)) for k, v in scores.items()), key=lambda kv: kv[0])
    xs = [k for k, _ in items]
    ys = [v for _, v in items]

    best = payload.get("best_window_size")
    metric = payload.get("metric", "score")

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Window size")
    plt.ylabel(metric)
    if best is not None:
        plt.title(f"Window-size selection (best={best})")
    else:
        plt.title("Window-size selection")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=160)
    print(f"Saved {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
