#!/usr/bin/env python3
"""
Plot confusion matrix from ts-agents evaluate_windowed_classifier[_from_csv] output.

Usage:
  python demo/plot_confusion_matrix.py outputs/demo/eval.json outputs/demo/confusion_matrix.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _extract_labels(payload: Dict[str, Any]) -> List[str]:
    # Best-effort: class_counts keys are label names.
    counts = payload.get("class_counts")
    if isinstance(counts, dict) and counts:
        return sorted([str(k) for k in counts.keys()])
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json", type=str)
    ap.add_argument("out_png", type=str)
    args = ap.parse_args()

    payload: Dict[str, Any] = json.loads(Path(args.eval_json).read_text())
    payload = payload.get("result", payload)
    clf = payload.get("classification") or {}
    cm = clf.get("confusion_matrix")

    if cm is None:
        raise ValueError("No classification.confusion_matrix found in eval JSON.")

    cm_arr = np.array(cm, dtype=float)
    labels = _extract_labels(payload)
    if labels and len(labels) != cm_arr.shape[0]:
        # Fallback: don't label if mismatch
        labels = []

    plt.figure()
    plt.imshow(cm_arr)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if labels:
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)

    # Annotate counts
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            plt.text(j, i, str(int(cm_arr[i, j])), ha="center", va="center")

    plt.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=160)
    print(f"Saved {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
