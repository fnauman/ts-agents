"""Tests for the M4 Monthly mini-panel example workflow helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_example_module():
    module_path = Path("examples/forecasting_m4_monthly_mini.py")
    spec = importlib.util.spec_from_file_location(
        "forecasting_m4_monthly_mini",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smape_treats_zero_zero_pairs_as_zero_contribution():
    module = _load_example_module()

    actual = np.array([0.0, 2.0])
    forecast = np.array([0.0, 4.0])

    # The zero/zero term should contribute 0, not be dropped from the average.
    assert module._smape(actual, forecast) == pytest.approx(100.0 / 3.0)


def test_load_panel_accepts_reordered_columns(tmp_path):
    module = _load_example_module()
    dataset_path = tmp_path / "m4_reordered.csv"

    pd.DataFrame(
        [
            {"split": "train", "unique_id": "M4", "y": 1.0, "ds": 1, "note": "x"},
            {"split": "train", "unique_id": "M4", "y": 2.0, "ds": 2, "note": "x"},
            {"split": "holdout", "unique_id": "M4", "y": 3.0, "ds": 3, "note": "x"},
        ]
    ).to_csv(dataset_path, index=False, columns=["split", "unique_id", "y", "ds", "note"])

    panel = module._load_panel(dataset_path, ["M4"])

    np.testing.assert_allclose(panel["M4"]["train"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(panel["M4"]["holdout"], np.array([3.0]))
    np.testing.assert_array_equal(panel["M4"]["holdout_ds"], np.array([3]))


def test_load_panel_rejects_missing_required_columns(tmp_path):
    module = _load_example_module()
    dataset_path = tmp_path / "m4_missing.csv"

    pd.DataFrame(
        [
            {"unique_id": "M4", "split": "train", "ds": 1},
        ]
    ).to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Missing required dataset columns"):
        module._load_panel(dataset_path, ["M4"])
