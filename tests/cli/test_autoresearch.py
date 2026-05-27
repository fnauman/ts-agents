import json
import subprocess

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


def test_autoresearch_run_foundation_gpu_plan_materializes_plan_only_recipes(capsys, tmp_path):
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
    assert payload["result"]["data"]["plan"]["target_hardware"] == "1x RTX PRO 6000 Blackwell"
    assert payload["result"]["data"]["plan"]["plan_only"] is True
    assert payload["result"]["data"]["plan"]["model_revisions"]["amazon/chronos-2"]
    assert (output_dir / "foundation_gpu_plan.json").exists()
    assert (output_dir / "chronos_finetune_config.yaml").exists()
    assert (output_dir / "moment_classification_config.yaml").exists()
    assert (output_dir / "commands.sh").exists()

    subprocess.run(["bash", "-n", str(output_dir / "commands.sh")], check=True)
    chronos_config = yaml.safe_load((output_dir / "chronos_finetune_config.yaml").read_text())
    moment_config = yaml.safe_load((output_dir / "moment_classification_config.yaml").read_text())
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
