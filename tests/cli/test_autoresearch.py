import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
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
    assert "foundation-smoke" in names
    assert "foundation-gpu-plan" in names
    # Deprecated aliases resolve but are never listed.
    assert "foundation-chronos-smoke" not in names


def test_autoresearch_show_resolves_deprecated_foundation_alias(capsys):
    code = run(["autoresearch", "show", "foundation-chronos-smoke", "--json"])

    assert code == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["name"] == "foundation-smoke"
    assert result["default_models"] == ["chronos2_small"]
    assert result["models"] == ["chronos2_small", "chronos2", "timesfm2p5", "patchtst_fm"]
    assert result["required_extras"] == ["foundation"]
    capabilities = result["capabilities"]
    assert capabilities["context_length"] == 512
    assert set(capabilities["checkpoints"]) == set(result["models"])
    assert (
        capabilities["checkpoints"]["chronos2_small"]["hub_model_name"]
        == "autogluon/chronos-2-small"
    )


def test_autoresearch_leases_unpublished_output_and_final_manifest(
    tmp_path, monkeypatch, capsys,
):
    import ts_agents.autoresearch.executor as executor_module
    import ts_agents.cli.main as cli_module
    from ts_agents.cli.runs import gc_runs
    from ts_agents.workflows.lifecycle import lock_run_output

    outer = tmp_path / "outputs" / "outer"
    outer.mkdir(parents=True)
    (outer / "run_manifest.json").write_text(json.dumps({
        "workflow": "inspect-series", "run_id": "outer", "status": "failed",
        "created_at": "2026-05-01T12:00:00Z",
    }))
    child = outer / "child"
    original_execute = executor_module.execute_autoresearch
    original_sync = cli_module._synchronize_autoresearch_manifest
    stages = []

    def execute(*args, **kwargs):
        assert not child.exists()
        result = gc_runs(tmp_path / "outputs", apply=True)
        assert result["runs"] == []
        assert "busy" in result["skipped"][0]["reason"]
        with pytest.raises(ValueError, match="busy"):
            with lock_run_output(outer):
                pytest.fail("parent overwrite acquired the active child's lease")
        stages.append("execute")
        return original_execute(*args, **kwargs)

    def synchronize(*args, **kwargs):
        with pytest.raises(ValueError, match="busy"):
            with lock_run_output(child):
                pytest.fail("output lease released before host manifest update")
        stages.append("synchronize")
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(executor_module, "execute_autoresearch", execute)
    monkeypatch.setattr(cli_module, "_synchronize_autoresearch_manifest", synchronize)
    assert run([
        "autoresearch", "run", "forecast-daytona", "--profile", "smoke",
        "--models", "seasonal_naive", "--max-trials", "1", "--skip-plots",
        "--output-dir", str(child), "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert stages == ["execute", "synchronize"]
    assert (child / "run_manifest.json").is_file()
    with lock_run_output(child):
        pass  # The lease is released after successful completion.


@pytest.mark.parametrize("busy_scope", ["same", "parent", "child", "home"])
def test_autoresearch_refuses_conflicting_output_lease(
    tmp_path, monkeypatch, capsys, busy_scope,
):
    import ts_agents.autoresearch.executor as executor_module
    from ts_agents.workflows.lifecycle import lock_run_output

    output = tmp_path / "output"
    scopes = {"same": output, "parent": tmp_path, "child": output / "nested", "home": output}
    monkeypatch.setenv("HOME", str(tmp_path))
    output_arg = "~/output" if busy_scope == "home" else str(output)

    def unexpected_execute(*_args, **_kwargs):
        pytest.fail("execution started despite a conflicting output lease")

    monkeypatch.setattr(executor_module, "execute_autoresearch", unexpected_execute)
    with lock_run_output(scopes[busy_scope]):
        code = run([
            "autoresearch", "run", "forecast-daytona", "--dry-run",
            "--output-dir", output_arg, "--json",
        ])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert "busy" in payload["error"]["message"]
    assert not output.exists()


def test_autoresearch_releases_output_lease_on_failure(tmp_path, monkeypatch, capsys):
    import ts_agents.autoresearch.executor as executor_module
    from ts_agents.workflows.lifecycle import lock_run_output

    output = tmp_path / "failed"

    def fail_execute(*_args, **_kwargs):
        with pytest.raises(ValueError, match="busy"):
            with lock_run_output(output):
                pytest.fail("execution was not protected by an output lease")
        raise ValueError("controlled execution failure")

    monkeypatch.setattr(executor_module, "execute_autoresearch", fail_execute)
    assert run([
        "autoresearch", "run", "forecast-daytona", "--dry-run",
        "--output-dir", str(output), "--json",
    ]) == 2
    assert "controlled execution failure" in json.loads(capsys.readouterr().out)["error"]["message"]
    with lock_run_output(output):
        pass


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


def _fake_foundation_backend(monkeypatch):
    """Stub the Darts adapter seams so no darts, torch or weights are needed."""
    from ts_agents.core.forecasting import catalog, foundation

    resolved = []
    constructed = []

    class FakePrediction:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=float).reshape(-1, 1)

        def values(self, copy=False):
            return self._values

    class FakeSeries:
        def __init__(self, values):
            self.values = np.asarray(values)

        @classmethod
        def from_times_and_values(cls, _times, values, columns=None):
            return cls(values)

    class FakeModel:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.horizon = kwargs["output_chunk_length"]

        def fit(self, _series):
            return self

        def predict(self, n, series, verbose=False):
            return [FakePrediction(np.full(n, float(item.values[-1]))) for item in series]

    def fake_import_model_class(spec):
        resolved.append(spec.darts_class)
        return FakeModel

    monkeypatch.setattr(catalog, "missing_modules", lambda _name: [])
    monkeypatch.setattr(foundation, "_import_model_class", fake_import_model_class)
    monkeypatch.setattr(foundation, "_timeseries_cls", lambda: FakeSeries)
    return resolved, constructed


def test_autoresearch_run_foundation_smoke_dry_run_writes_contract(capsys, tmp_path):
    output_dir = tmp_path / "foundation-dry-run"
    code = run(
        [
            "autoresearch",
            "run",
            "foundation-smoke",
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
    assert payload["result"]["status"] == "ok"
    assert payload["result"]["warnings"] == []
    assert payload["result"]["data"]["trial_count"] == 1
    assert payload["result"]["data"]["best_config"] == {}
    assert (output_dir / "trials.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "run_manifest.json").exists()

    from ts_agents.autoresearch.registry import (
        FOUNDATION_SMOKE_MODEL_SCOPE,
        FOUNDATION_SMOKE_MODEL_SCOPE_LABEL,
    )

    report = (output_dir / "report.md").read_text()
    assert report.startswith("# Foundation Model Smoke (Darts)")
    assert FOUNDATION_SMOKE_MODEL_SCOPE_LABEL in report
    assert "autogluon/chronos-2-small" in report
    assert "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a" in report
    summary = json.loads((output_dir / "summary.json").read_text())
    assert list(summary["checkpoints"]) == ["chronos2_small"]
    # Run outputs must not reference repo-only files that are not shipped
    # in the wheel.
    assert "external_benchmark_context" not in summary
    assert "benchmarks/" not in report
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["loop"] == "foundation-smoke"
    assert manifest["options"]["models"] == ["chronos2_small"]
    assert manifest["options"]["context_length"] == 512
    assert manifest["options"]["model_scope"] == FOUNDATION_SMOKE_MODEL_SCOPE
    assert manifest["options"]["model_scope_label"] == FOUNDATION_SMOKE_MODEL_SCOPE_LABEL
    trial = json.loads((output_dir / "trials.jsonl").read_text().splitlines()[0])
    assert trial["model"] == "chronos2_small"
    assert trial["status"] == "planned"


def test_autoresearch_foundation_smoke_dry_run_imports_no_heavy_modules(tmp_path):
    output_dir = tmp_path / "isolation"
    script = (
        "import json, sys\n"
        "from ts_agents.cli.main import run\n"
        "code = run(['autoresearch', 'run', 'foundation-smoke', '--dry-run', "
        f"'--skip-plots', '--output-dir', {str(output_dir)!r}, '--json'])\n"
        "heavy = sorted({'darts', 'torch', 'huggingface_hub', 'pytorch_lightning'} "
        "& set(sys.modules))\n"
        "print(json.dumps({'code': code, 'heavy': heavy}), file=sys.stderr)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stderr.strip().splitlines()[-1])
    assert result == {"code": 0, "heavy": []}


def test_autoresearch_run_foundation_smoke_executes_only_selected_model(
    monkeypatch, capsys, tmp_path
):
    import ts_agents.autoresearch.executor as executor_module
    from ts_agents.core.forecasting import foundation

    monkeypatch.setattr(executor_module, "find_spec", lambda _name: object())
    resolved, constructed = _fake_foundation_backend(monkeypatch)
    output_dir = tmp_path / "foundation-timesfm"
    code = run(
        [
            "autoresearch",
            "run",
            "foundation-smoke",
            "--models",
            "timesfm2p5",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["result"]["status"] == "ok"
    data = payload["result"]["data"]
    assert data["trial_count"] == 1
    assert data["model_scope"] == "darts_foundation_zero_shot_smoke"
    assert data["best_config"]["model"] == "timesfm2p5"
    assert data["best_config"]["smape"] is not None
    # Only the selected Darts class is resolved and constructed, with its pin.
    assert resolved == ["TimesFM2p5Model"]
    assert len(constructed) == 1
    assert constructed[0]["hub_model_name"] == "google/timesfm-2.5-200m-pytorch"
    assert constructed[0]["hub_model_revision"] == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    assert constructed[0]["input_chunk_length"] == (1, 512)
    assert constructed[0]["output_chunk_length"] == 18
    # The run-scoped model cache is released when the loop finishes.
    assert foundation._MODEL_CACHE == {}

    trial = json.loads((output_dir / "trials.jsonl").read_text().splitlines()[0])
    assert trial["trial_id"] == "foundation-001-timesfm2p5"
    assert trial["task"] == "foundation-model-smoke"
    assert trial["execution_mode"] == "zero_shot"
    assert trial["foundation_family"] == "TimesFM"
    assert trial["darts_class"] == "TimesFM2p5Model"
    assert trial["hub_model_revision"] == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    assert "season_length" not in (output_dir / "trials.csv").read_text()
    report = (output_dir / "report.md").read_text()
    assert "google/timesfm-2.5-200m-pytorch" in report
    assert "chronos-2" not in report


def test_autoresearch_foundation_smoke_defaults_to_one_chronos2_small_trial(
    monkeypatch, capsys, tmp_path
):
    import ts_agents.autoresearch.executor as executor_module

    monkeypatch.setattr(executor_module, "find_spec", lambda _name: object())
    resolved, _constructed = _fake_foundation_backend(monkeypatch)
    output_dir = tmp_path / "foundation-full"
    code = run(
        [
            "autoresearch",
            "run",
            "foundation-smoke",
            "--profile",
            "full",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["result"]["data"]["trial_count"] == 1
    assert payload["result"]["data"]["best_config"]["model"] == "chronos2_small"
    assert resolved == ["Chronos2Model"]
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["options"]["models"] == ["chronos2_small"]
    assert manifest["options"]["max_trials"] == 4


def test_autoresearch_foundation_smoke_deprecated_alias_runs_new_loop(
    monkeypatch, capsys, tmp_path
):
    import ts_agents.autoresearch.executor as executor_module

    monkeypatch.setattr(executor_module, "find_spec", lambda _name: object())
    _fake_foundation_backend(monkeypatch)
    output_dir = tmp_path / "foundation-alias"
    code = run(
        [
            "autoresearch",
            "run",
            "foundation-chronos-smoke",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    # A deprecation notice asks for review but does not degrade the run.
    assert payload["result"]["status"] == "ok"
    warnings = payload["result"]["warnings"]
    assert any(
        "'foundation-chronos-smoke' is deprecated" in warning
        and "'foundation-smoke'" in warning
        for warning in warnings
    )
    assert payload["result"]["data"]["loop"] == "foundation-smoke"
    assert payload["result"]["data"]["best_config"]["model"] == "chronos2_small"
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["loop"] == "foundation-smoke"
    assert manifest["warnings"] == warnings


def test_autoresearch_alias_default_output_dir_uses_canonical_name(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.chdir(tmp_path)
    code = run(["autoresearch", "run", "foundation-chronos-smoke", "--dry-run", "--json"])

    assert code == 0
    output_dir = Path(json.loads(capsys.readouterr().out)["result"]["data"]["output_dir"])
    assert output_dir.parent == (tmp_path / "outputs" / "autoresearch" / "foundation-smoke").resolve()


def test_autoresearch_foundation_smoke_rejects_unknown_model(capsys, tmp_path):
    code = run(
        [
            "autoresearch",
            "run",
            "foundation-smoke",
            "--models",
            "not-a-model",
            "--dry-run",
            "--output-dir",
            str(tmp_path / "foundation-unknown"),
            "--json",
        ]
    )

    assert code == 2
    message = json.loads(capsys.readouterr().out)["error"]["message"]
    assert "Unsupported model(s) for foundation-smoke: not-a-model" in message
    assert "chronos2_small" in message


def test_normalize_models_uses_loop_default_models():
    from ts_agents.autoresearch.runner import _normalize_models

    assert _normalize_models("foundation-smoke", None) == ["chronos2_small"]
    assert _normalize_models("foundation-smoke", ["", "  "]) == ["chronos2_small"]
    assert _normalize_models("foundation-chronos-smoke", None) == ["chronos2_small"]
    assert _normalize_models("foundation-smoke", ["patchtst_fm", "chronos2"]) == [
        "patchtst_fm",
        "chronos2",
    ]
    # Loops without default_models still run every model.
    assert _normalize_models("forecast-daytona", None) == [
        "seasonal_naive",
        "theta",
        "ets",
        "arima",
    ]


def test_autoresearch_foundation_smoke_preflight_reports_missing_deps(
    monkeypatch, tmp_path
):
    import ts_agents.autoresearch.executor as executor_module
    from ts_agents.autoresearch.executor import AutoresearchExecutor
    from ts_agents.tools.executor import ExecutionContext, SandboxMode, ToolErrorCode

    def fake_find_spec(name):
        if name == "darts":
            return None
        return object()

    monkeypatch.setattr(executor_module, "find_spec", fake_find_spec)
    result = AutoresearchExecutor().execute(
        "foundation-smoke",
        {"output_dir": str(tmp_path / "foundation-deps")},
        context=ExecutionContext(sandbox_mode=SandboxMode.LOCAL),
    )

    assert not result.success
    assert result.error is not None
    assert result.error.code == ToolErrorCode.DEPENDENCY_ERROR
    assert "darts" in result.error.message
    assert "ts-agents[foundation]" in result.error.message
    assert not (tmp_path / "foundation-deps").exists()


def test_autoresearch_foundation_smoke_failed_trial_degrades(
    monkeypatch, capsys, tmp_path
):
    import ts_agents.autoresearch.executor as executor_module
    import ts_agents.autoresearch.runner as runner_module

    monkeypatch.setattr(executor_module, "find_spec", lambda _name: object())

    def unavailable(_model, _series, *, horizon, context_length):
        from ts_agents.core.forecasting.foundation import FoundationModelUnavailableError

        raise FoundationModelUnavailableError("weights are not cached")

    monkeypatch.setattr(runner_module, "_forecast_with_foundation", unavailable)
    code = run(
        [
            "autoresearch",
            "run",
            "foundation-smoke",
            "--skip-plots",
            "--output-dir",
            str(tmp_path / "foundation-failed"),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["status"] == "degraded"
    assert payload["result"]["data"]["best_config"] == {}
    trial = json.loads(
        (tmp_path / "foundation-failed" / "trials.jsonl").read_text().splitlines()[0]
    )
    assert trial["status"] == "failed"
    assert trial["error_type"] == "FoundationModelUnavailableError"


def test_autoresearch_foundation_smoke_fails_empty_holdout():
    from ts_agents.autoresearch.runner import _evaluate_forecast_trial_with_runner

    row = _evaluate_forecast_trial_with_runner(
        run_id="run",
        trial_index=1,
        spec={
            "series_id": "M4",
            "phase": "holdout",
            "origin": 3,
            "train": np.array([1.0, 2.0, 3.0]),
            "actual": np.array([]),
        },
        model="chronos2_small",
        task="foundation-model-smoke",
        trial_id_prefix="foundation",
        forecast_runner=lambda _model, _series, *, horizon: np.ones(horizon),
        horizon=18,
    )

    assert row["status"] == "failed"
    assert "no actual holdout values" in row["error"]


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
    plan = payload["result"]["data"]["plan"]
    assert plan["target_hardware"] == "1x RTX PRO 6000 Blackwell"
    assert plan["plan_only"] is True
    assert (
        plan["model_revisions"]["amazon/chronos-2"]
        == "29ec3766d36d6f73f0696f85560a422f50e8498c"
    )
    assert set(plan["model_revisions"]) == {
        "amazon/chronos-2",
        "google/timesfm-2.5-200m-pytorch",
        "ibm-granite/granite-timeseries-patchtst-fm-r1",
        "AutonLab/MOMENT-1-large",
    }
    assert plan["forecasting"]["backend"] == "darts"
    assert plan["forecasting"]["package"] == "darts[torch]>=0.47,<0.48"
    assert set(plan["forecasting"]["zero_shot_comparators"]) == {"timesfm2p5", "patchtst_fm"}
    assert plan["classification"]["support"] == "external_plan_only"
    runs = plan["recommended_runs"]
    assert [(row.get("method"), row["mode"]) for row in runs] == [
        ("chronos2", "zero_shot_evaluation"),
        ("chronos2", "fine_tune"),
        ("timesfm2p5", "zero_shot_comparator"),
        ("patchtst_fm", "zero_shot_comparator"),
        (None, "linear_probe_then_peft_after_adapter"),
    ]
    assert runs[1]["darts_kwargs"] == {"enable_finetuning": True}
    assert runs[-1]["support"] == "external_plan_only"
    assert not any(".arrow" in item for item in plan["adapter_requirements"])

    assert (output_dir / "foundation_gpu_plan.json").exists()
    assert (output_dir / "darts_finetune_config.yaml").exists()
    assert not (output_dir / "chronos_finetune_config.yaml").exists()
    assert (output_dir / "moment_classification_config.yaml").exists()
    assert (output_dir / "commands.sh").exists()

    subprocess.run(["bash", "-n", str(output_dir / "commands.sh")], check=True)
    darts_config = yaml.safe_load(
        (output_dir / "darts_finetune_config.yaml").read_text()
    )
    moment_config = yaml.safe_load(
        (output_dir / "moment_classification_config.yaml").read_text()
    )
    assert darts_config["darts_class"] == "Chronos2Model"
    assert darts_config["hub_model_name"] == "amazon/chronos-2"
    assert darts_config["hub_model_revision"] == "29ec3766d36d6f73f0696f85560a422f50e8498c"
    assert darts_config["enable_finetuning"] is True
    assert darts_config["input_chunk_length"] == 512
    assert darts_config["output_chunk_length"] == 18
    assert darts_config["optimizer_kwargs"]["lr"] > 0
    assert darts_config["pl_trainer_kwargs"] == {
        "accelerator": "gpu",
        "devices": 1,
        "precision": "bf16-mixed",
    }
    assert darts_config["data"]["split"] == "train"
    assert darts_config["adapter_required"] is False
    assert moment_config["adapter_required"] is True
    assert moment_config["dataset_path"] is None
    commands = (output_dir / "commands.sh").read_text()
    assert "training/train.py" not in commands
    assert "ts-agents[foundation]" in commands
    assert "chronos" not in commands.lower()
    assert ".arrow" not in commands
    assert "windowed_activity_dataset.npz" in commands
    report = (output_dir / "report.md").read_text()
    assert "darts_finetune_config.yaml" in report
    assert "external_plan_only" in report

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["loop"] == "foundation-gpu-plan"
    assert manifest["status"] == "plan-only"
    assert manifest["best_config"] == {}


def test_foundation_gpu_plan_registry_uses_catalog_pins():
    from ts_agents.autoresearch.registry import get_loop
    from ts_agents.core.forecasting.catalog import get_method

    loop = get_loop("foundation-gpu-plan")
    chronos2 = get_method("chronos2").foundation
    assert loop.models == ["chronos2", "timesfm2p5", "patchtst_fm", "AutonLab/MOMENT-1-large"]
    capabilities = loop.capabilities
    assert capabilities["forecasting_backend"] == "darts"
    assert capabilities["forecasting_model"] == chronos2.hub_model_name
    assert capabilities["forecasting_model_revision"] == chronos2.hub_model_revision
    assert capabilities["forecasting_finetune"] == "Chronos2Model(enable_finetuning=True)"
    assert capabilities["zero_shot_comparators"] == ["timesfm2p5", "patchtst_fm"]
    assert capabilities["classification_support"].startswith("external_plan_only:")
    assert "forecasting_finetune_fallback" not in capabilities


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
    foundation_context = _context_with_loop_environment(
        context, get_loop("foundation-smoke"), SandboxMode.DAYTONA
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
    assert (
        foundation_context.environment["TS_AGENTS_DAYTONA_INSTALL_EXTRAS"]
        == "foundation"
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


def test_foundation_smoke_uses_and_records_configured_context(monkeypatch, tmp_path):
    from ts_agents.autoresearch import runner
    from ts_agents.autoresearch.registry import get_loop
    from dataclasses import replace
    definition = get_loop("foundation-smoke")
    configured = replace(definition, capabilities={**definition.capabilities, "context_length": 37})
    monkeypatch.setattr(runner, "get_loop", lambda name: configured)
    observed = []
    def forecast(model, series, *, horizon, context_length):
        observed.append(context_length)
        return np.full(horizon, series[-1])
    monkeypatch.setattr(runner, "_forecast_with_foundation", forecast)
    result = runner.run_autoresearch_loop(loop_name="foundation-smoke", output_dir=str(tmp_path),
        models=["chronos2_small"], skip_plots=True)
    assert observed == [37]
    trial = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[0])
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert trial["context_length"] == manifest["options"]["context_length"] == 37
    assert manifest["options"]["resolved_context_length"] == {"chronos2_small": 37}
    assert result["status"] == "ok"


def test_deprecated_alias_writes_notice_with_initial_manifest(monkeypatch, tmp_path):
    from ts_agents.autoresearch import runner
    writes = []
    original = runner.write_output
    def record(content, path):
        if Path(path).name == "run_manifest.json":
            writes.append(json.loads(content))
        return original(content, path)
    monkeypatch.setattr(runner, "write_output", record)
    result = runner.run_autoresearch_loop(loop_name="foundation-chronos-smoke", output_dir=str(tmp_path),
        dry_run=True, skip_plots=True)
    assert len(writes) == 1
    assert writes[0]["warnings"] == result["warnings"]
    assert "deprecated" in writes[0]["warnings"][0].lower()
    assert result["status"] == "ok"
