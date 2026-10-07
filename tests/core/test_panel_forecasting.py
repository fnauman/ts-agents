"""Input, chronological split and prediction coverage contracts for panel models."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ts_agents.cli.input_parsing import PanelInput, load_panel_input
from ts_agents.core.forecasting.panel import normalize_panel, score_predictions
from ts_agents.workflows.lifecycle import resume_identity
from ts_agents.workflows.panel import run_forecast_panel_workflow


def panel_frame(length=60):
    return pd.DataFrame(
        [
            {"unique_id": name, "ds": date, "y": float(10 + shift + i % 7 + i / 10)}
            for name, shift in (("a", 0), ("b", 10))
            for i, date in enumerate(
                pd.date_range("2024-01-01", periods=length, freq="D")
            )
        ]
    )


def panel_input(frame=None):
    frame = panel_frame() if frame is None else frame.copy()
    frame["ds"] = frame.ds.map(lambda date: date.isoformat())
    return PanelInput(frame.to_dict("records"), "inline_json", "test")


@pytest.mark.parametrize(
    "problem",
    ["gap", "duplicate", "nan", "infinity", "unaligned", "extra", "negative_freq"],
)
def test_panel_rejects_ambiguous_data(problem):
    frame = panel_frame()
    freq = "D"
    if problem == "gap":
        frame = frame.drop(5)
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "nan":
        frame.loc[0, "y"] = np.nan
    elif problem == "infinity":
        frame.loc[0, "y"] = np.inf
    elif problem == "unaligned":
        frame = frame.drop(frame.index[-1])
    elif problem == "extra":
        frame["weather"] = 1
    else:
        freq = "-1D"
    with pytest.raises(ValueError):
        normalize_panel(frame, freq)


def test_panel_serialization_and_resume_include_all_rows():
    from ts_agents.workflows.executor import (
        _deserialize_workflow_input,
        _serialize_workflow_input,
    )

    data = panel_input()
    restored = _deserialize_workflow_input(_serialize_workflow_input(data))
    assert restored == data
    identity = resume_identity("forecast-panel", data, {"freq": "D"})
    restored.records[0]["y"] += 1
    assert resume_identity("forecast-panel", restored, {"freq": "D"}) != identity


def test_custom_columns_and_csv_input(tmp_path):
    path = tmp_path / "panel.csv"
    panel_frame().rename(
        columns={"unique_id": "customer", "ds": "time", "y": "load"}
    ).to_csv(path, index=False)
    result = load_panel_input(
        input_path=str(path),
        id_col="customer",
        time_col="time",
        value_col="load",
        freq="D",
    )
    assert len(result.records) == 120
    assert set(result.records[0]) == {"unique_id", "ds", "y"}


def test_json_filename_may_start_with_bracket(monkeypatch, tmp_path):
    from ts_agents.cli.input_parsing import load_json_value

    monkeypatch.chdir(tmp_path)
    Path("[panel].json").write_text('{"value": 1}')
    assert load_json_value(input_json="[panel].json") == ({"value": 1}, "json_file")


def test_backtests_use_only_pre_cutoff_data_and_save_artifacts(monkeypatch, tmp_path):
    from ts_agents.workflows import panel as workflow
    from ts_agents.core.forecasting.panel import PanelBackend

    original = PanelBackend.fit
    fit_ends = []

    def capture(self, frame, horizon):
        fit_ends.append(frame.ds.max())
        original(self, frame, horizon)

    monkeypatch.setattr(workflow.PanelBackend, "fit", capture)
    frame = panel_frame()
    output = run_forecast_panel_workflow(
        panel_input(frame),
        output_dir=str(tmp_path),
        freq="D",
        horizon=3,
        methods=["seasonal_naive"],
        season_length=7,
        n_windows=2,
        skip_plots=True,
    )
    predictions = pd.read_csv(
        tmp_path / "backtest_predictions.csv", parse_dates=["ds", "cutoff"]
    )
    assert (predictions.ds > predictions.cutoff).all()
    assert predictions.groupby(["model", "window", "unique_id"]).size().eq(3).all()
    assert fit_ends == [
        frame.ds.max() - pd.Timedelta(days=6),
        frame.ds.max() - pd.Timedelta(days=3),
        frame.ds.max(),
    ]
    assert (tmp_path / "models/seasonal_naive/history.csv").exists()
    assert len(pd.read_csv(tmp_path / "forecast.csv")) == 6
    assert output.data["evaluation"].startswith("rolling validation")
    assert all(Path(artifact.path).is_file() for artifact in output.artifacts)


def test_train_only_mase_and_prediction_coverage():
    frame = panel_frame(30)
    train = frame.groupby("unique_id").head(27)
    actual = frame.groupby("unique_id").tail(3)
    predicted = actual.rename(columns={"y": "model"}).copy()
    predicted["model"] += 1
    scored = score_predictions(actual, predicted, "model", train, 7)
    assert scored.mase_scale.to_numpy() == pytest.approx(np.full(6, 0.7))
    for broken in (predicted.iloc[1:], pd.concat([predicted, predicted.iloc[:1]])):
        with pytest.raises(ValueError):
            score_predictions(actual, broken, "model", train, 7)


def test_zero_scale_is_reported_without_nan_json(tmp_path):
    frame = panel_frame()
    frame["y"] = 1.0
    output = run_forecast_panel_workflow(
        panel_input(frame),
        output_dir=str(tmp_path),
        freq="D",
        horizon=3,
        methods=["seasonal_naive"],
        season_length=7,
        n_windows=1,
        skip_plots=True,
    )
    assert output.status == "degraded"
    assert output.data["metrics"][0]["mase"] is None
    assert "NaN" not in (tmp_path / "metrics.json").read_text()


def test_missing_dependencies_fail_before_training(monkeypatch, tmp_path):
    from ts_agents.workflows import panel

    monkeypatch.setattr(panel, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"ts-agents\[ml\]"):
        run_forecast_panel_workflow(
            panel_input(),
            output_dir=str(tmp_path),
            freq="D",
            horizon=3,
            methods=["lightgbm"],
            season_length=7,
            lags=[1, 7],
            n_windows=1,
        )


def test_cli_import_and_discovery_do_not_import_trainers():
    script = """
import sys
from ts_agents.workflows import get_workflow
assert get_workflow("forecast-panel").name == "forecast-panel"
assert not {"torch", "lightgbm", "mlforecast", "neuralforecast"}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


def test_panel_cli_discovery_and_run(capsys, tmp_path):
    from ts_agents.cli.main import run

    assert run(["workflow", "show", "forecast-panel", "--json"]) == 0
    metadata = json.loads(capsys.readouterr().out)["result"]
    assert metadata["capabilities"]["supported_methods"] == [
        "seasonal_naive",
        "lightgbm",
        "histgbm",
        "nhits",
    ]
    data = panel_input()
    args = [
        "workflow",
        "run",
        "forecast-panel",
        "--input-json",
        json.dumps(data.records),
        "--freq",
        "D",
        "--horizon",
        "3",
        "--season-length",
        "7",
        "--n-windows",
        "1",
        "--methods",
        "seasonal_naive",
        "--skip-plots",
        "--output-dir",
        str(tmp_path / "run"),
        "--json",
    ]
    assert run(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"]
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert manifest["resume_identity"]["input_sha256"]
    data.records[0]["y"] += 1
    args[args.index("--input-json") + 1] = json.dumps(data.records)
    assert run([*args, "--resume"]) != 0
    capsys.readouterr()


@pytest.mark.parametrize("method", ["lightgbm", "histgbm", "nhits"])
def test_real_optional_backend_training_persistence_and_reload(method, tmp_path):
    """Small real fits, never mocked: native saved models reproduce future forecasts."""
    if method == "nhits":
        pytest.importorskip("neuralforecast")
        from neuralforecast import NeuralForecast
    else:
        pytest.importorskip("mlforecast")
        if method == "lightgbm":
            pytest.importorskip("lightgbm")
        from mlforecast import MLForecast
    from ts_agents.core.forecasting.panel import PanelBackend

    config = dict(
        lags=[1, 7],
        n_estimators=5,
        max_steps=3,
        input_size=6,
        num_threads=1,
        accelerator="cpu",
        seed=1337,
    )
    frame = normalize_panel(panel_frame(35), "D")
    backend = PanelBackend(method, "D", 7, config)
    backend.fit(frame, 3)
    expected = backend.predict(3).sort_values(["unique_id", "ds"])
    assert len(expected) == 6 and np.isfinite(expected[method]).all()
    backend.save(tmp_path / method)
    if method == "nhits":
        restored = NeuralForecast.load(str(tmp_path / method))
        actual = restored.predict()
    else:
        restored = MLForecast.load(str(tmp_path / method))
        actual = restored.predict(3)
    actual = actual.sort_values(["unique_id", "ds"])
    assert actual[method].to_numpy() == pytest.approx(
        expected[method].to_numpy(), rel=1e-6, abs=1e-6
    )
    # Reruns/resumes save into the same directory; the newest fit replaces it.
    (tmp_path / method / "stale.bin").write_bytes(b"old")
    backend.save(tmp_path / method)
    assert not (tmp_path / method / "stale.bin").exists()


def test_save_replaces_existing_model_directory_without_leftovers(tmp_path):
    from ts_agents.core.forecasting.panel import PanelBackend

    backend = PanelBackend("seasonal_naive", "D", 7, {})
    backend.fit(normalize_panel(panel_frame(20), "D"), 3)
    target = tmp_path / "models" / "seasonal_naive"
    target.mkdir(parents=True)
    (target / "stale.ckpt").write_text("previous run")
    backend.save(target)
    backend.save(target)
    assert sorted(path.name for path in (tmp_path / "models").iterdir()) == ["seasonal_naive"]
    assert sorted(path.name for path in target.iterdir()) == ["history.csv"]


def test_panel_subprocess_keeps_artifacts_and_json_envelope(tmp_path):
    path = tmp_path / "input.csv"
    panel_frame().to_csv(path, index=False)
    # Real command from a new Python process exercises panel serialization.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ts_agents",
            "workflow",
            "run",
            "forecast-panel",
            "--input",
            str(path),
            "--freq",
            "D",
            "--horizon",
            "3",
            "--season-length",
            "7",
            "--n-windows",
            "1",
            "--methods",
            "seasonal_naive",
            "--skip-plots",
            "--sandbox",
            "subprocess",
            "--output-dir",
            str(tmp_path / "subprocess"),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"]
    assert all(
        Path(artifact["path"]).is_file() for artifact in payload["result"]["artifacts"]
    )
