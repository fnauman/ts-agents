import json
import os
import signal
import subprocess
import time

import pytest
import yaml

from ts_agents.cli.main import run


def test_autoresearch_list_json_returns_loops(capsys):
    code = run(["autoresearch", "list", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "autoresearch list"
    names = [loop["name"] for loop in payload["result"]["loops"]]
    assert "forecast-daytona" in names
    assert "classify-daytona" in names
    assert "foundation-gpu-plan" in names


def test_autoresearch_show_json_returns_budget(capsys):
    code = run(["autoresearch", "show", "forecast-daytona", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "forecast-daytona"
    result = payload["result"]
    assert result["primary_metric"] == "smape"
    assert result["budget"]["vcpu"] == 4
    assert "seasonal_naive" in result["models"]


def test_autoresearch_run_rejects_zero_max_trials(capsys, tmp_path):
    code = run(
        [
            "autoresearch",
            "run",
            "forecast-daytona",
            "--profile",
            "smoke",
            "--models",
            "seasonal_naive",
            "--max-trials",
            "0",
            "--output-dir",
            str(tmp_path / "forecast-zero"),
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "--max-trials must be a positive integer" in payload["error"]["message"]


def test_autoresearch_run_forecast_smoke_writes_artifacts(capsys, tmp_path):
    output_dir = tmp_path / "forecast"
    code = run(
        [
            "autoresearch",
            "run",
            "forecast-daytona",
            "--profile",
            "smoke",
            "--models",
            "seasonal_naive",
            "--max-trials",
            "2",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["quality_status"] == "ok"
    assert payload["execution"]["backend_actual"] == "local"
    assert payload["result"]["data"]["trial_count"] == 2
    assert payload["result"]["data"]["best_config"]["model"] == "seasonal_naive"
    assert payload["result"]["data"]["best_config"]["n_trials"] == 2
    assert payload["result"]["data"]["best_config"]["n_holdout_trials"] == 1
    assert payload["result"]["data"]["best_config"]["n_rolling_trials"] == 1
    assert (output_dir / "trials.csv").exists()
    assert (output_dir / "trials.jsonl").exists()
    assert (output_dir / "best_config.json").exists()
    assert (output_dir / "run_manifest.json").exists()
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["loop"] == "forecast-daytona"
    assert manifest["best_config"]["model"] == "seasonal_naive"
    assert manifest["best_config"]["n_trials"] == 2
    assert manifest["options"]["max_trials"] == 2


def test_autoresearch_manifest_includes_plot_artifact(capsys, tmp_path):
    output_dir = tmp_path / "forecast-plot"
    code = run(
        [
            "autoresearch",
            "run",
            "forecast-daytona",
            "--profile",
            "smoke",
            "--models",
            "seasonal_naive",
            "--max-trials",
            "1",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert (output_dir / "ranking.png").exists()
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    manifest_paths = {artifact["path"] for artifact in manifest["artifacts"]}
    assert str(output_dir / "ranking.png") in manifest_paths
    assert str(output_dir / "run_manifest.json") not in manifest_paths


def test_autoresearch_run_classification_dry_run(capsys, tmp_path):
    output_dir = tmp_path / "classification"
    code = run(
        [
            "autoresearch",
            "run",
            "classify-daytona",
            "--profile",
            "smoke",
            "--models",
            "knn",
            "--dataset",
            "synthetic",
            "--max-trials",
            "1",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["result"]["data"]["trial_count"] == 1
    assert (output_dir / "synthetic_labeled_stream.csv").exists()
    assert (output_dir / "trials.csv").exists()


def test_autoresearch_run_foundation_gpu_plan_materializes_plan_only_recipes(
    capsys, tmp_path
):
    output_dir = tmp_path / "foundation"
    code = run(
        [
            "autoresearch",
            "run",
            "foundation-gpu-plan",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["quality_status"] == "review"
    assert payload["result"]["status"] == "plan-only"
    assert payload["result"]["data"]["trial_count"] == 0
    assert payload["result"]["data"]["best_config"] == {}
    assert (
        payload["result"]["data"]["plan"]["target_hardware"]
        == "1x RTX PRO 6000 Blackwell"
    )
    assert payload["result"]["data"]["plan"]["plan_only"] is True
    assert payload["result"]["data"]["plan"]["model_revisions"]["amazon/chronos-2"]
    assert (output_dir / "foundation_gpu_plan.json").exists()
    assert (output_dir / "chronos_finetune_config.yaml").exists()
    assert (output_dir / "moment_classification_config.yaml").exists()
    assert (output_dir / "commands.sh").exists()

    subprocess.run(["bash", "-n", str(output_dir / "commands.sh")], check=True)
    chronos_config = yaml.safe_load(
        (output_dir / "chronos_finetune_config.yaml").read_text()
    )
    moment_config = yaml.safe_load(
        (output_dir / "moment_classification_config.yaml").read_text()
    )
    assert chronos_config["adapter_required"] is True
    assert chronos_config["prediction_length"] == 18
    assert chronos_config["training_data_paths"] == []
    assert moment_config["adapter_required"] is True
    assert moment_config["dataset_path"] is None
    assert "training/train.py" not in (output_dir / "commands.sh").read_text()

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["loop"] == "foundation-gpu-plan"
    assert manifest["status"] == "plan-only"
    assert manifest["best_config"] == {}


def test_autoresearch_subprocess_sandbox_hook(capsys, tmp_path):
    output_dir = tmp_path / "subprocess"
    code = run(
        [
            "autoresearch",
            "run",
            "forecast-daytona",
            "--profile",
            "smoke",
            "--models",
            "seasonal_naive",
            "--max-trials",
            "1",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--sandbox",
            "subprocess",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["execution"]["backend_actual"] == "subprocess"
    assert payload["result"]["data"]["output_dir"] == str(output_dir)


def test_autoresearch_unknown_loop_json_error(capsys):
    code = run(["autoresearch", "show", "not-a-loop", "--json"])

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["command"] == "autoresearch show"
    assert payload["name"] == "not-a-loop"
    assert payload["error"]["code"] == "validation_error"
    assert "Unknown autoresearch loop" in payload["error"]["message"]


def test_autoresearch_serialized_runner_honors_artifact_dir_env(monkeypatch, tmp_path):
    from ts_agents.autoresearch.executor import _run_serialized_autoresearch

    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("TS_AGENTS_TOOL_ARTIFACT_DIR", str(artifact_root))

    result = _run_serialized_autoresearch(
        loop_name="forecast-daytona",
        options={
            "profile": "smoke",
            "models": ["seasonal_naive"],
            "max_trials": 1,
            "dry_run": True,
            "skip_plots": True,
        },
        use_sandbox_artifact_dir=True,
    )

    assert result["data"]["output_dir"] == str(artifact_root / "forecast-daytona")
    assert (artifact_root / "forecast-daytona" / "run_manifest.json").exists()


def test_autoresearch_artifact_limit_falls_back_after_invalid_specific_env(monkeypatch):
    from ts_agents.autoresearch.executor import _autoresearch_artifact_bundle_limits

    monkeypatch.setenv("TS_AGENTS_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES", "not-an-int")
    monkeypatch.setenv("TS_AGENTS_WORKFLOW_ARTIFACT_MAX_FILE_BYTES", "123")
    monkeypatch.setenv("TS_AGENTS_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES", "also-bad")
    monkeypatch.setenv("TS_AGENTS_WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES", "456")

    assert _autoresearch_artifact_bundle_limits() == (123, 456)


def test_forecast_ranking_puts_nan_metrics_last():
    from ts_agents.autoresearch.runner import _rank_forecast_trials

    ranking = _rank_forecast_trials(
        [
            {
                "status": "ok",
                "model": "bad",
                "phase": "holdout",
                "smape": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
            },
            {
                "status": "ok",
                "model": "good",
                "phase": "holdout",
                "smape": 1.0,
                "mae": 1.0,
                "rmse": 1.0,
            },
        ]
    )

    assert [row["model"] for row in ranking] == ["good", "bad"]
    assert ranking[1]["smape"] is None


def test_autoresearch_preserves_explicit_zero_timeout(capsys, tmp_path):
    output_dir = tmp_path / "zero-timeout"
    code = run(
        [
            "autoresearch",
            "run",
            "forecast-daytona",
            "--profile",
            "smoke",
            "--models",
            "seasonal_naive",
            "--max-trials",
            "1",
            "--timeout-seconds",
            "0",
            "--dry-run",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["options"]["timeout_seconds"] == 0


def test_autoresearch_cli_preserves_explicit_zero_resource_context(
    monkeypatch, capsys, tmp_path
):
    import ts_agents.autoresearch.executor as executor_module
    from ts_agents.tools.executor import ExecutionResult, ExecutionStatus

    captured = {}

    def fake_execute_autoresearch(_loop_name, _options, *, context):
        captured["context"] = context
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            result={
                "kind": "autoresearch",
                "summary": "ok",
                "status": "ok",
                "data": {
                    "output_dir": str(tmp_path / "out"),
                    "manifest_path": str(tmp_path / "out" / "run_manifest.json"),
                    "best_config": {},
                },
                "artifacts": [],
                "warnings": [],
            },
        )

    monkeypatch.setattr(
        executor_module, "execute_autoresearch", fake_execute_autoresearch
    )
    code = run(
        [
            "autoresearch",
            "run",
            "forecast-daytona",
            "--timeout-seconds",
            "0",
            "--memory-mb",
            "0",
            "--disk-mb",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert captured["context"].timeout_seconds == 0
    assert captured["context"].memory_mb == 0
    assert captured["context"].disk_mb == 0


def test_forecast_full_profile_expands_dry_run_trials(capsys, tmp_path):
    counts = {}
    for profile in ["default", "full"]:
        output_dir = tmp_path / profile
        code = run(
            [
                "autoresearch",
                "run",
                "forecast-daytona",
                "--profile",
                profile,
                "--models",
                "seasonal_naive",
                "--dry-run",
                "--skip-plots",
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        counts[profile] = payload["result"]["data"]["trial_count"]

    assert counts["full"] > counts["default"]


def test_autoresearch_plot_failure_keeps_manifest(monkeypatch, capsys, tmp_path):
    from ts_agents.autoresearch import runner

    def raise_plot(_output_path, _trials):
        raise OSError("read-only output")

    monkeypatch.setattr(runner, "_write_forecast_plot", raise_plot)
    output_dir = tmp_path / "plot-failure"
    code = run(
        [
            "autoresearch",
            "run",
            "forecast-daytona",
            "--profile",
            "smoke",
            "--models",
            "seasonal_naive",
            "--max-trials",
            "1",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality_status"] == "degraded"
    assert any(
        "Skipped ranking plot" in warning for warning in payload["result"]["warnings"]
    )
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "trials.csv").exists()


def test_normalize_models_rejects_null_entries():
    from ts_agents.autoresearch.runner import _normalize_models

    with pytest.raises(ValueError, match="contains a null entry"):
        _normalize_models("classify-daytona", [None, "knn"])


def test_classification_ranking_puts_missing_metric_after_real_zero():
    from ts_agents.autoresearch.runner import _rank_classification_trials

    ranking = _rank_classification_trials(
        [
            {
                "status": "ok",
                "model": "missing",
                "best_window_size": 32,
                "n_windows": 10,
            },
            {
                "status": "ok",
                "model": "zero",
                "balanced_accuracy": 0.0,
                "best_window_size": 64,
                "n_windows": 10,
            },
        ]
    )

    assert [row["model"] for row in ranking] == ["zero", "missing"]


def test_daytona_extras_context_isolated_between_loops():
    from ts_agents.autoresearch.executor import _context_with_loop_environment
    from ts_agents.autoresearch.registry import get_loop
    from ts_agents.tools.executor import ExecutionContext, SandboxMode

    context = ExecutionContext(
        sandbox_mode=SandboxMode.DAYTONA,
        environment={"TS_AGENTS_DAYTONA_INSTALL_EXTRAS": "previous"},
    )
    forecast_context = _context_with_loop_environment(
        context, get_loop("forecast-daytona"), SandboxMode.DAYTONA
    )
    classify_context = _context_with_loop_environment(
        context, get_loop("classify-daytona"), SandboxMode.DAYTONA
    )

    assert context.environment == {"TS_AGENTS_DAYTONA_INSTALL_EXTRAS": "previous"}
    assert (
        forecast_context.environment["TS_AGENTS_DAYTONA_INSTALL_EXTRAS"]
        == "forecasting"
    )
    assert (
        classify_context.environment["TS_AGENTS_DAYTONA_INSTALL_EXTRAS"]
        == "classification"
    )


def test_autoresearch_local_dependency_preflight(monkeypatch, tmp_path):
    import ts_agents.autoresearch.executor as executor_module
    from ts_agents.autoresearch.executor import AutoresearchExecutor
    from ts_agents.tools.executor import ExecutionContext, SandboxMode, ToolErrorCode

    def fake_find_spec(name):
        if name == "statsforecast":
            return None
        return object()

    monkeypatch.setattr(executor_module, "find_spec", fake_find_spec)
    result = AutoresearchExecutor().execute(
        "forecast-daytona",
        {
            "output_dir": str(tmp_path / "deps"),
            "models": ["theta"],
        },
        context=ExecutionContext(sandbox_mode=SandboxMode.LOCAL),
    )

    assert not result.success
    assert result.error is not None
    assert result.error.code == ToolErrorCode.DEPENDENCY_ERROR
    assert "statsforecast" in result.error.message


def test_docker_artifact_materialization_preserves_nested_paths(tmp_path):
    from ts_agents.autoresearch.executor import _materialize_existing_artifacts
    from ts_agents.tools.executor import ExecutionResult, ExecutionStatus

    source_root = tmp_path / "sandbox"
    plot = source_root / "plots" / "ranking.png"
    data = source_root / "data" / "summary.json"
    plot.parent.mkdir(parents=True)
    data.parent.mkdir(parents=True)
    plot.write_bytes(b"plot")
    data.write_text("{}")
    result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        result={
            "data": {
                "output_dir": str(source_root),
                "manifest_path": str(source_root / "run_manifest.json"),
            },
            "artifacts": [
                {"path": str(plot), "kind": "image"},
                {"path": str(data), "kind": "json"},
            ],
        },
    )
    destination = tmp_path / "host"

    _materialize_existing_artifacts(result, str(destination))

    assert (destination / "plots" / "ranking.png").read_bytes() == b"plot"
    assert (destination / "data" / "summary.json").read_text() == "{}"
    assert not (destination / "ranking.png").exists()


def test_remote_artifact_materialization_rejects_symlink_destination(tmp_path):
    from ts_agents.autoresearch.executor import (
        _STAGED_AUTORESEARCH_ARTIFACTS_KEY,
        _materialize_remote_autoresearch_output,
    )
    from ts_agents.tools.executor import ExecutionResult, ExecutionStatus

    if not hasattr(os, "symlink"):
        pytest.skip("symlink support is required")
    destination = tmp_path / "host"
    destination.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    try:
        os.symlink(outside, destination / "evil")
    except OSError:
        pytest.skip("symlink creation is not permitted")

    result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        result={
            "data": {"manifest_path": "run_manifest.json"},
            "artifacts": [{"path": "/sandbox/evil", "kind": "text"}],
            _STAGED_AUTORESEARCH_ARTIFACTS_KEY: [
                {
                    "source_path": "/sandbox/evil",
                    "relative_path": "evil",
                    "content_base64": "b3ZlcndyaXRl",
                }
            ],
        },
    )

    _materialize_remote_autoresearch_output(result, str(destination))

    assert outside.read_text() == "keep"
    assert any(
        "destination is a symlink" in warning
        for warning in result.result.get("warnings", [])
    )


def test_docker_artifact_dir_is_run_scoped():
    from ts_agents.autoresearch.executor import _sandbox_autoresearch_artifact_dir
    from ts_agents.tools.executor import SandboxMode

    first = _sandbox_autoresearch_artifact_dir(SandboxMode.DOCKER)
    second = _sandbox_autoresearch_artifact_dir(SandboxMode.DOCKER)

    assert first != second
    assert first.startswith("/io/artifacts/")
    assert second.startswith("/io/artifacts/")


def test_trial_timeout_restores_expired_outer_alarm_immediately():
    from ts_agents.autoresearch.runner import _trial_timeout

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "ITIMER_REAL"):
        pytest.skip("SIGALRM timers are required")

    class OuterAlarm(Exception):
        pass

    def raise_outer_alarm(_signum, _frame):
        raise OuterAlarm

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, raise_outer_alarm)
    signal.setitimer(signal.ITIMER_REAL, 0.05)
    try:
        with pytest.raises(OuterAlarm):
            with _trial_timeout(1.0):
                time.sleep(0.1)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
