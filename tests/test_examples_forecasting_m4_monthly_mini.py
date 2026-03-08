"""Tests for the M4 Monthly mini-panel example workflow helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

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


def test_run_workflow_reduced_profile_writes_expected_artifacts(tmp_path):
    module = _load_example_module()

    output_dir = tmp_path / "forecasting-workflow-smoke"
    result = module.run_workflow(
        output_dir=output_dir,
        series_ids=["M4", "M100"],
        methods=["seasonal_naive", "theta"],
        horizon=18,
        season_length=12,
        rolling_origins=2,
    )

    artifacts = result["artifacts"]
    metrics_path = artifacts["metrics_by_series"]
    summary_path = artifacts["summary"]
    forecasts_path = artifacts["holdout_forecasts"]
    report_path = artifacts["report"]
    run_summary_path = artifacts["run_summary"]
    plot_paths = artifacts["plots"]

    assert metrics_path.exists()
    assert summary_path.exists()
    assert forecasts_path.exists()
    assert report_path.exists()
    assert run_summary_path.exists()
    assert len(plot_paths) >= 1
    assert all(path.exists() and path.stat().st_size > 0 for path in plot_paths)

    metrics_df = pd.read_csv(metrics_path)
    summary_df = pd.read_csv(summary_path)
    forecasts_df = pd.read_csv(forecasts_path)
    run_summary = json.loads(run_summary_path.read_text())
    report = report_path.read_text()

    assert list(metrics_df.columns) == ["phase", "origin", "unique_id", "method", "smape", "mae", "rmse"]
    assert set(metrics_df["phase"]) == {"rolling_origin", "holdout"}
    assert set(metrics_df["unique_id"]) == {"M4", "M100"}
    assert set(metrics_df["method"]) == {"seasonal_naive", "theta"}
    assert len(metrics_df) == 12

    assert list(summary_df.columns) == ["phase", "method", "smape", "mae", "rmse"]
    assert len(summary_df) == 4
    assert set(summary_df["phase"]) == {"rolling_origin", "holdout"}
    assert set(summary_df["method"]) == {"seasonal_naive", "theta"}

    assert list(forecasts_df.columns) == ["unique_id", "method", "ds", "actual", "forecast"]
    assert len(forecasts_df) == 72
    assert set(forecasts_df["unique_id"]) == {"M4", "M100"}
    assert set(forecasts_df["method"]) == {"seasonal_naive", "theta"}

    assert run_summary["series"] == ["M4", "M100"]
    assert run_summary["methods"] == ["seasonal_naive", "theta"]
    assert run_summary["horizon"] == 18
    assert run_summary["season_length"] == 12
    assert run_summary["rolling_origins"] == 2
    assert run_summary["best_method"] in {"seasonal_naive", "theta"}
    assert Path(run_summary["artifacts"]["metrics_by_series"]) == metrics_path
    assert Path(run_summary["artifacts"]["summary"]) == summary_path
    assert Path(run_summary["artifacts"]["holdout_forecasts"]) == forecasts_path
    assert Path(run_summary["artifacts"]["report"]) == report_path
    assert Path(run_summary["artifacts"]["run_summary"]) == run_summary_path
    assert [Path(path) for path in run_summary["artifacts"]["plots"]] == plot_paths

    holdout_summary = summary_df[summary_df["phase"] == "holdout"].reset_index(drop=True)
    assert holdout_summary.iloc[0]["method"] == run_summary["best_method"]

    assert "# Professional Forecasting Workflow Report" in report
    assert "## Holdout ranking" in report
    assert f"- recommended method: `{run_summary['best_method']}`" in report
    assert "metrics_by_series.csv" in report
    assert "holdout_forecasts.csv" in report
    assert "plots/M4.png" in report


def test_main_treats_empty_plot_series_as_default(monkeypatch, tmp_path, capsys):
    module = _load_example_module()
    observed = {}

    def fake_run_workflow(**kwargs):
        observed.update(kwargs)
        output_dir = kwargs["output_dir"]
        summary_path = output_dir / "summary.csv"
        return {
            "output_dir": output_dir,
            "best_method": "theta",
            "artifacts": {"summary": summary_path},
        }

    monkeypatch.setattr(module, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "forecasting_m4_monthly_mini.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--plot-series",
            "",
        ],
    )

    module.main()

    assert observed["plot_series"] is None
    output = capsys.readouterr().out
    assert "Best holdout method: theta" in output
