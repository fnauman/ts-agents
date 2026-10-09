import base64
import argparse
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ts_agents.cli.main import run
from ts_agents.cli.input_parsing import SeriesInput
from ts_agents.core.comparison import ComparisonResult


def test_workflow_list_json_returns_envelope(capsys):
    code = run(["workflow", "list", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1.0"
    assert payload["ok"] is True
    assert payload["command"] == "workflow list"
    workflow_names = [workflow["name"] for workflow in payload["result"]["workflows"]]
    assert "activity-recognition" in workflow_names
    assert "inspect-series" in workflow_names
    assert "forecast-series" in workflow_names
    assert "forecast-panel" in workflow_names


def test_workflow_show_forecast_panel_reports_foundation_models(capsys):
    from ts_agents.core.forecasting.catalog import foundation_methods, methods_for

    assert run(["workflow", "show", "forecast-panel", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    capabilities = result["capabilities"]
    assert capabilities["foundation_models"] is True
    assert capabilities["zero_shot_methods"] == foundation_methods("panel")
    assert capabilities["method_extras"]["chronos2_small"] == "foundation"
    assert capabilities["method_extras"]["seasonal_naive"] is None
    checkpoint = capabilities["foundation_model_checkpoints"]["chronos2_small"]
    assert checkpoint["hub_model_name"] == "autogluon/chronos-2-small"
    assert result["availability"]["status"] == "available"
    features = {feature["name"]: feature for feature in result["availability"]["optional_features"]}
    assert features["foundation_models"]["required_extras"] == ["foundation"]
    options = {option["name"]: option for option in result["options"]}
    assert options["methods"]["choices"] == methods_for("panel")
    assert "context_length" in options
    assert any("chronos2_small" in example for example in result["examples"])


def test_workflow_show_forecast_series_lists_foundation_methods(capsys):
    from ts_agents.core.forecasting.catalog import methods_for

    assert run(["workflow", "show", "forecast-series", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["capabilities"]["supported_methods"] == methods_for("series")
    assert result["capabilities"]["foundation_models"] is True
    availability = result["availability"]
    features = {feature["name"] for feature in availability["optional_features"]}
    assert {"foundation_models", "plots"} <= features
    # Missing foundation models never change forecast-series status on their own.
    expected = "available" if "statsforecast" not in availability["missing_dependencies"] else "degraded"
    assert availability["status"] == expected
    options = {option["name"] for option in result["options"]}
    assert {"context_length", "accelerator"} <= options


def test_workflow_show_json_returns_machine_metadata(capsys):
    code = run(["workflow", "show", "forecast-series", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "workflow show"
    assert payload["name"] == "forecast-series"
    result = payload["result"]
    assert result["supported_input_modes"] == [
        "input_file",
        "input_json",
        "stdin",
        "bundled_run",
    ]
    assert "artifacts" in result
    assert "availability" in result
    assert "supported_methods" in result["capabilities"]
    assert "seasonal_naive" in result["capabilities"]["supported_methods"]
    assert "source_options" in result
    assert "global_options" in result
    assert "status_contract" in result
    assert "cli_templates" in result
    assert "default_output_behavior" in result
    source_option_names = [option["name"] for option in result["source_options"]]
    assert "input" in source_option_names
    assert "input_json" in source_option_names
    assert "run_id" in source_option_names
    assert result["default_output_behavior"]["default_output_dir"] == "outputs/forecast"
    global_option_names = [option["name"] for option in result["global_options"]]
    assert "overwrite" in global_option_names
    assert "resume" in global_option_names
    assert result["default_output_behavior"]["manifest_filename"] == "run_manifest.json"
    assert result["default_output_behavior"]["supports_resume"] is True
    assert any(
        template == "ts-agents workflow show forecast-series --json"
        for template in result["cli_templates"]
    )


def test_unknown_workflow_error_uses_generic_hint(capsys):
    code = run(["workflow", "show", "forecast-seriez"])

    assert code == 2
    err = capsys.readouterr().err
    assert "workflow list" in err
    assert "workflow show inspect-series --json" not in err


def test_workflow_run_inspect_series_accepts_stdin_json(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"series":[1,2,3,4,5]}'))

    output_dir = tmp_path / "inspect"
    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--stdin",
            "--output-dir",
            str(output_dir),
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["quality_status"] == "ok"
    assert payload["degraded"] is False
    assert payload["requires_review"] is False
    assert payload["name"] == "inspect-series"
    assert payload["result"]["data"]["workflow"] == "inspect-series"
    assert payload["result"]["data"]["run_id"]
    assert payload["result"]["data"]["manifest_path"] == str(output_dir / "run_manifest.json")
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "ledger.json").exists()
    assert (output_dir / "report.md").exists()
    assert payload["result"]["data"]["forecast_recommendation"]["choice"]
    artifact_paths = {artifact["path"] for artifact in payload["result"]["artifacts"]}
    assert str(output_dir / "summary.json") in artifact_paths
    assert str(output_dir / "ledger.json") in artifact_paths
    assert str(output_dir / "report.md") in artifact_paths
    assert str(output_dir / "run_manifest.json") in artifact_paths
    manifest_payload = json.loads((output_dir / "run_manifest.json").read_text())
    manifest_artifact_paths = {artifact["path"] for artifact in manifest_payload["artifacts"]}
    assert str(output_dir / "ledger.json") in manifest_artifact_paths
    assert str(output_dir / "run_manifest.json") in manifest_artifact_paths
    assert payload["result"]["data"]["execution"]["backend_requested"] == "local"
    assert payload["result"]["data"]["execution"]["backend_actual"] == "local"
    assert payload["result"]["data"]["run"]["execution"]["fallback_used"] is False
    assert manifest_payload["execution"]["backend_requested"] == "local"
    assert manifest_payload["execution"]["backend_actual"] == "local"
    assert manifest_payload["execution"]["fallback_allowed"] is False
    assert manifest_payload["execution"]["fallback_used"] is False
    assert payload["result"]["data"]["autocorrelation"]["max_lag"] == 4
    assert payload["result"]["data"]["autocorrelation"]["requested_max_lag"] == 8


def test_workflow_run_manifest_sync_write_failure_is_reported(monkeypatch, capsys, tmp_path):
    import importlib

    cli_main = importlib.import_module("ts_agents.cli.main")
    from ts_agents.workflows import lifecycle
    original_write_manifest = lifecycle.write_manifest

    def flaky_write_manifest(path, manifest):
        if manifest["status"] == "ok":
            raise OSError("disk full")
        return original_write_manifest(path, manifest)

    monkeypatch.setattr(lifecycle, "write_manifest", flaky_write_manifest)

    output_dir = tmp_path / "inspect"
    code = cli_main.run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            str(output_dir),
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 6
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["message"] == "disk full"
    assert json.loads((output_dir / "run_manifest.json").read_text())["status"] == "failed"


def test_workflow_run_inspect_series_supports_subprocess_sandbox(capsys, tmp_path):
    output_dir = tmp_path / "inspect_subprocess"
    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            str(output_dir),
            "--skip-plots",
            "--sandbox",
            "subprocess",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["execution"]["backend_requested"] == "subprocess"
    assert payload["execution"]["backend_actual"] == "subprocess"
    assert payload["result"]["data"]["output_dir"] == str(output_dir.resolve())
    assert payload["result"]["data"]["manifest_path"] == str((output_dir / "run_manifest.json").resolve())
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "ledger.json").exists()
    assert (output_dir / "report.md").exists()


def test_workflow_run_inspect_series_restricts_recommendation_to_available_methods(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.workflows.inspect as inspect_mod

    monkeypatch.setattr(inspect_mod, "_available_forecasting_methods", lambda: ["seasonal_naive"])

    output_dir = tmp_path / "inspect_base_profile"
    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5,6,7,8,9,10,11,12]}',
            "--output-dir",
            str(output_dir),
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    recommendation = payload["result"]["data"]["forecast_recommendation"]
    assert recommendation["choice"] == "seasonal_naive"
    assert recommendation["alternatives"] == []
    ledger_payload = json.loads((output_dir / "ledger.json").read_text())
    recommendation_entry = next(
        entry for entry in ledger_payload["entries"] if entry["key"] == "forecasting_method"
    )
    assert recommendation_entry["value"] == "seasonal_naive"


def test_workflow_run_generates_run_scoped_output_dir_by_default(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)

    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = Path(payload["result"]["data"]["output_dir"])
    assert output_dir.parent == (tmp_path / "outputs" / "inspect").resolve()
    assert output_dir.name == payload["result"]["data"]["run_id"]
    assert (output_dir / "run_manifest.json").exists()


def test_workflow_run_rejects_nonempty_explicit_output_dir_without_overwrite_or_resume(capsys, tmp_path):
    output_dir = tmp_path / "inspect"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("stale")

    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            str(output_dir),
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"
    assert "Use --overwrite" in payload["error"]["message"]


def test_workflow_run_overwrite_clears_explicit_output_dir(capsys, tmp_path):
    output_dir = tmp_path / "inspect"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("stale")

    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            str(output_dir),
            "--overwrite",
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["data"]["run"]["resumed"] is False
    assert not (output_dir / "stale.txt").exists()
    assert (output_dir / "run_manifest.json").exists()


def test_workflow_run_overwrite_does_not_clear_output_dir_when_late_validation_fails(capsys, tmp_path):
    output_dir = tmp_path / "inspect"
    output_dir.mkdir()
    stale_path = output_dir / "stale.txt"
    stale_path.write_text("stale")

    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            str(output_dir),
            "--overwrite",
            "--fallback-backend",
            "local",
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "--allow-fallback" in payload["error"]["message"]
    assert stale_path.exists()


def test_workflow_run_resume_reuses_manifest_run_id(capsys, tmp_path):
    output_dir = tmp_path / "inspect"

    first_code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            str(output_dir),
            "--skip-plots",
            "--json",
        ]
    )
    assert first_code == 0
    first_payload = json.loads(capsys.readouterr().out)
    first_run_id = first_payload["result"]["data"]["run_id"]

    second_code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            str(output_dir),
            "--resume",
            "--skip-plots",
            "--json",
        ]
    )

    assert second_code == 0
    second_payload = json.loads(capsys.readouterr().out)
    assert second_payload["result"]["data"]["run_id"] == first_run_id
    assert second_payload["result"]["data"]["run"]["resumed"] is True


@pytest.mark.parametrize("change", ["workflow", "input", "options", "legacy"])
def test_workflow_resume_rejects_incompatible_run_without_modifying_evidence(capsys, tmp_path, change):
    output_dir = tmp_path / "run"
    argv = ["workflow", "run", "inspect-series", "--input-json",
            '{"series":[1,2,3,4,5]}', "--output-dir", str(output_dir), "--skip-plots", "--json"]
    assert run(argv) == 0
    capsys.readouterr()
    manifest_path = output_dir / "run_manifest.json"
    if change == "legacy":
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("resume_identity")
        manifest_path.write_text(json.dumps(manifest))
    before = {path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()}
    resumed = argv + ["--resume"]
    if change == "workflow":
        resumed[2] = "forecast-series"
        resumed.extend(["--horizon", "2", "--methods", "seasonal_naive"])
    elif change == "input":
        resumed[4] = '{"series":[1,2,3,4,6]}'
    elif change == "options":
        resumed.extend(["--max-lag", "2"])
    assert run(resumed) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "validation_error"
    after = {path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()}
    assert after == before


def test_workflow_resume_detects_changed_file_contents(capsys, tmp_path):
    source = tmp_path / "data.json"
    source.write_text('{"series":[1,2,3,4,5]}')
    argv = ["workflow", "run", "inspect-series", "--input-json", str(source),
            "--output-dir", str(tmp_path / "run"), "--skip-plots", "--json"]
    assert run(argv) == 0
    capsys.readouterr()
    before = (tmp_path / "run" / "run_manifest.json").read_bytes()
    source.write_text('{"series":[1,2,3,4,6]}')
    assert run(argv + ["--resume"]) == 2
    capsys.readouterr()
    assert (tmp_path / "run" / "run_manifest.json").read_bytes() == before


def test_workflow_catalogs_active_and_failed_execution(capsys, tmp_path, monkeypatch):
    from ts_agents.workflows import executor as executor_module
    from ts_agents.cli.runs import gc_runs, scan_runs

    output_dir = tmp_path / "outputs" / "run"

    def fail_execution(*args, **kwargs):
        records, warnings = scan_runs(tmp_path / "outputs")
        assert warnings == []
        assert records[0]["status"] == "running"
        assert gc_runs(tmp_path / "outputs", apply=True)["runs"] == []
        assert run(["workflow", "run", "inspect-series", "--input-json", '{"series":[1,2,3,4,5]}',
                    "--output-dir", str(output_dir), "--overwrite", "--skip-plots", "--json"]) == 2
        assert json.loads((output_dir / "run_manifest.json").read_text())["status"] == "running"
        raise RuntimeError("controlled execution failure")

    monkeypatch.setattr(executor_module, "execute_workflow", fail_execution)
    assert run(["workflow", "run", "inspect-series", "--input-json", '{"series":[1,2,3,4,5]}',
                "--output-dir", str(output_dir), "--skip-plots", "--json"]) == 6
    capsys.readouterr()
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["message"] == "controlled execution failure"
    assert manifest["resume_identity"]["input_sha256"]


def test_parent_overwrite_refuses_unpublished_nested_writer(tmp_path, capsys):
    from ts_agents.workflows.lifecycle import lock_run_output

    outer = tmp_path / "parent"
    outer.mkdir()
    evidence = outer / "keep.txt"
    evidence.write_text("keep")
    with lock_run_output(outer / "child"):
        assert run(["workflow", "run", "inspect-series", "--input-json", '{"series":[1,2,3,4,5]}',
                    "--output-dir", str(outer), "--overwrite", "--skip-plots", "--json"]) == 2
        assert "busy" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert evidence.read_text() == "keep"


@pytest.mark.skipif(os.name != "posix", reason="shared ancestor leases on POSIX")
def test_independent_sibling_output_leases_can_coexist(tmp_path):
    from ts_agents.workflows.lifecycle import lock_run_output

    with lock_run_output(tmp_path / "first"), lock_run_output(tmp_path / "second"):
        pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and no-follow leases")
@pytest.mark.parametrize("poison", ["directory-link", "public-directory", "lease-link", "lease-hardlink"])
def test_output_leases_refuse_poisoned_paths(tmp_path, monkeypatch, poison):
    import hashlib
    from ts_agents.workflows import lifecycle

    monkeypatch.setattr(lifecycle.tempfile, "gettempdir", lambda: str(tmp_path))
    name = hashlib.sha256(str(os.getuid()).encode()).hexdigest()[:16]
    root = tmp_path / f"ts-agents-run-locks-{name}"
    target = tmp_path / "untouched"
    target.write_bytes(b"")
    if poison == "directory-link":
        real = tmp_path / "other-directory"
        real.mkdir(mode=0o700)
        root.symlink_to(real, target_is_directory=True)
    else:
        root.mkdir(mode=0o700)
        if poison == "public-directory":
            root.chmod(0o777)
        else:
            lease = root / (hashlib.sha256(b"/").hexdigest() + ".lease")
            if poison == "lease-link":
                lease.symlink_to(target)
            else:
                os.link(target, lease)
    with pytest.raises((ValueError, OSError)):
        with lifecycle.lock_run_output(tmp_path / "run"):
            pytest.fail("poisoned lease was accepted")
    assert target.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory ownership")
def test_output_leases_refuse_wrong_owner(tmp_path, monkeypatch):
    from ts_agents.workflows import lifecycle

    monkeypatch.setattr(lifecycle.tempfile, "gettempdir", lambda: str(tmp_path))
    fstat = os.fstat
    def wrong_owner(fd):
        value = fstat(fd)
        return SimpleNamespace(st_uid=value.st_uid + 1, st_mode=value.st_mode)
    monkeypatch.setattr(lifecycle.os, "fstat", wrong_owner)
    with pytest.raises(ValueError, match="owned by the current user"):
        with lifecycle.lock_run_output(tmp_path / "run"):
            pytest.fail("foreign lease directory was accepted")


def test_intermediate_manifest_write_failure_preserves_catalog(tmp_path, monkeypatch, capsys):
    from ts_agents.workflows import lifecycle

    dump = lifecycle.json.dump
    def fail_intermediate(value, handle, *args, **kwargs):
        if value.get("status") == "running" and "summary" in value:
            handle.write("{partial")
            raise OSError("controlled intermediate write failure")
        return dump(value, handle, *args, **kwargs)
    monkeypatch.setattr(lifecycle.json, "dump", fail_intermediate)
    output = tmp_path / "run"
    assert run(["workflow", "run", "inspect-series", "--input-json", '{"series":[1,2,3,4,5]}',
                "--output-dir", str(output), "--skip-plots", "--json"]) == 6
    capsys.readouterr()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["resume_identity"] and manifest["run_id"]
    assert "controlled intermediate write failure" in manifest["error"]["message"]


def test_persistent_status_write_error_preserves_original_exception(tmp_path, capsys, monkeypatch):
    from ts_agents.workflows import executor, lifecycle

    original_write = lifecycle.write_manifest
    def write_once(path, manifest):
        if Path(path).exists():
            raise OSError("status storage unavailable")
        original_write(path, manifest)
    monkeypatch.setattr(lifecycle, "write_manifest", write_once)
    def fail(*args, **kwargs):
        raise ValueError("original workflow failure")
    monkeypatch.setattr(executor, "execute_workflow", fail)
    output = tmp_path / "run"
    assert run(["workflow", "run", "inspect-series", "--input-json", '{"series":[1,2,3,4,5]}',
                "--output-dir", str(output), "--skip-plots", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["message"] == "original workflow failure"
    assert payload["error"]["details"]["status_recording_error"] == "status storage unavailable"
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "running"


def test_host_finalizes_manifest_when_remote_manifest_was_not_staged(tmp_path, capsys, monkeypatch):
    from ts_agents.workflows import executor
    from ts_agents.tools.executor import ExecutionResult, ExecutionStatus
    from ts_agents.contracts import ToolPayload

    def remote_result(*args, **kwargs):
        return ExecutionResult(status=ExecutionStatus.SUCCESS,
            result=ToolPayload(kind="workflow", summary="remote inspection", status="ok", data={
                "manifest_path": "/unavailable-sandbox/run_manifest.json",
                "output_dir": "/unavailable-sandbox", "source": {},
            }), metadata={"backend_actual": "daytona"})
    monkeypatch.setattr(executor, "execute_workflow", remote_result)
    output = tmp_path / "run"
    assert run(["workflow", "run", "inspect-series", "--input-json", '{"series":[1,2,3,4,5]}',
                "--output-dir", str(output), "--skip-plots", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["data"]["manifest_path"] == str(output / "run_manifest.json")
    assert result["artifacts"][-1]["path"] == str(output / "run_manifest.json")
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "ok" and manifest["resume_identity"]


@pytest.mark.parametrize("field", ["time_values", "labels", "provenance"])
def test_resume_fingerprint_preserves_nonfinite_distinctions(field):
    from ts_agents.workflows.lifecycle import resume_identity

    digests = []
    for value in (None, float("nan"), float("inf"), float("-inf")):
        source = SimpleNamespace(series=np.array([1., 2., 3.]))
        setattr(source, field, {"value": value} if field == "provenance" else [value])
        digests.append(resume_identity("inspect-series", source, {})["input_sha256"])
    assert len(set(digests)) == 4


def test_failed_resume_retains_creation_time(tmp_path, capsys, monkeypatch):
    from ts_agents.workflows import lifecycle

    output = tmp_path / "run"
    argv = ["workflow", "run", "inspect-series", "--input-json", '{"series":[1,2,3,4,5]}',
            "--output-dir", str(output), "--skip-plots", "--json"]
    assert run(argv) == 0
    capsys.readouterr()
    path = output / "run_manifest.json"
    original = json.loads(path.read_text())
    original["created_at"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(original))
    write = lifecycle.write_manifest
    def fail_finalization(path, manifest):
        if manifest["status"] == "ok":
            raise OSError("controlled finalization error")
        write(path, manifest)
    monkeypatch.setattr(lifecycle, "write_manifest", fail_finalization)
    assert run(argv + ["--resume"]) == 6
    capsys.readouterr()
    failed = json.loads(path.read_text())
    assert failed["status"] == "failed"
    assert failed["created_at"] == original["created_at"]
    assert failed["run_id"] == original["run_id"]


def test_invalid_sandbox_context_is_cataloged_as_failed(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TS_AGENTS_SANDBOX_MODE", "invalid-backend")
    output = tmp_path / "run"
    assert run(["workflow", "run", "inspect-series", "--input-json", '{"series":[1,2,3,4,5]}',
                "--output-dir", str(output), "--skip-plots", "--json"]) == 2
    capsys.readouterr()
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "failed"


def test_workflow_run_resume_rejects_invalid_manifest_json(capsys, tmp_path):
    output_dir = tmp_path / "inspect"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text("{not-json")

    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            str(output_dir),
            "--resume",
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "not valid JSON" in payload["error"]["message"]


def test_workflow_run_resume_rejects_non_object_manifest(capsys, tmp_path):
    output_dir = tmp_path / "inspect"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text("[]")

    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            str(output_dir),
            "--resume",
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "must contain a JSON object" in payload["error"]["message"]


def test_attach_workflow_run_metadata_rejects_non_dict_payload_data(tmp_path):
    from ts_agents.contracts import ToolPayload
    from ts_agents.workflows.common import attach_workflow_run_metadata

    payload = ToolPayload(
        kind="workflow",
        summary="ok",
        data=[],  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="payload.data must be a dict"):
        attach_workflow_run_metadata(
            payload,
            workflow_name="inspect-series",
            output_dir=tmp_path,
            run_id="run-123",
            source={},
            options={},
        )


def test_workflow_executor_skips_host_availability_gate_for_docker(monkeypatch):
    import ts_agents.workflows.executor as workflow_executor_mod
    from ts_agents.tools.executor import ExecutionContext, ExecutionResult, ExecutionStatus, SandboxMode

    backend_calls = {}
    probe_calls = {}

    class FakeDockerBackend:
        def is_available(self):
            return True

        def execute(self, tool_name, func, params, context):
            backend_calls["tool_name"] = tool_name
            backend_calls["params"] = params
            backend_calls["context"] = context
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                result={"kind": "workflow", "data": {}, "artifacts": []},
                formatted_output="ok",
                metadata={"backend": "docker"},
            )

    executor = workflow_executor_mod.WorkflowExecutor()
    executor.backends[SandboxMode.DOCKER] = FakeDockerBackend()

    monkeypatch.setattr(
        workflow_executor_mod,
        "get_workflow",
        lambda name: SimpleNamespace(
            availability=lambda: {
                "status": "unavailable",
                "available": False,
                "missing_dependencies": ["statsforecast"],
                "required_extras": ["forecasting"],
                "optional_features": [],
                "install_hint": "host is missing forecasting deps",
            }
        ),
    )
    monkeypatch.setattr(
        workflow_executor_mod,
        "describe_sandbox_backend",
        lambda mode, context=None, backend=None: probe_calls.update(
            {"mode": mode, "context": context, "backend": backend}
        ) or {
            "backend": mode.value,
            "available": True,
            "reason": None,
            "suggested_fix": None,
            "requirements": [],
            "details": {},
            "description": "backend",
        },
    )

    context = ExecutionContext(sandbox_mode="docker")
    result = executor.execute(
        "forecast-series",
        SeriesInput(series=np.array([1.0, 2.0, 3.0]), source_type="inline_json", label="series"),
        {"output_dir": "outputs/forecast", "horizon": 2, "methods": ["arima"]},
        context=context,
    )

    assert result.success is True
    assert backend_calls["tool_name"] == "workflow:forecast-series"
    assert probe_calls["mode"] == SandboxMode.DOCKER
    assert probe_calls["context"] is context
    assert probe_calls["backend"] is executor.backends[SandboxMode.DOCKER]


@pytest.mark.parametrize(
    ("workflow_name", "runner_kwargs", "expected"),
    [
        ("forecast-panel", {"methods": ["lightgbm", "chronos2"], "skip_plots": False}, "foundation,ml,viz"),
        ("forecast-panel", {"methods": ["seasonal_naive"], "skip_plots": True}, None),
        ("forecast-series", {"methods": ["seasonal_naive", "arima"], "skip_plots": False}, "forecasting,viz"),
        ("forecast-series", {"methods": ["chronos2_small"], "skip_plots": True}, "forecasting,foundation"),
    ],
)
def test_daytona_workflow_runs_install_extras_for_requested_methods(
    monkeypatch, workflow_name, runner_kwargs, expected
):
    from ts_agents.tools.executor import ExecutionContext, SandboxMode
    from ts_agents.workflows import get_workflow
    from ts_agents.workflows.executor import _context_with_workflow_extras

    monkeypatch.delenv("TS_AGENTS_DAYTONA_INSTALL_EXTRAS", raising=False)
    context = ExecutionContext(sandbox_mode=SandboxMode.DAYTONA, environment={"KEEP": "1"})
    updated = _context_with_workflow_extras(
        context, get_workflow(workflow_name), runner_kwargs, SandboxMode.DAYTONA
    )

    assert context.environment == {"KEEP": "1"}
    assert updated.environment.get("TS_AGENTS_DAYTONA_INSTALL_EXTRAS") == expected
    assert updated.environment["KEEP"] == "1"
    local = _context_with_workflow_extras(
        context, get_workflow(workflow_name), runner_kwargs, SandboxMode.LOCAL
    )
    assert local is context


def test_daytona_workflow_extras_preserve_user_setting(monkeypatch):
    from ts_agents.tools.executor import ExecutionContext, SandboxMode
    from ts_agents.workflows import get_workflow
    from ts_agents.workflows.executor import _context_with_workflow_extras

    workflow = get_workflow("forecast-panel")
    kwargs = {"methods": ["chronos2_small"], "skip_plots": False}
    monkeypatch.delenv("TS_AGENTS_DAYTONA_INSTALL_EXTRAS", raising=False)
    context = ExecutionContext(
        sandbox_mode=SandboxMode.DAYTONA,
        environment={"TS_AGENTS_DAYTONA_INSTALL_EXTRAS": "all"},
    )
    assert _context_with_workflow_extras(context, workflow, kwargs, SandboxMode.DAYTONA) is context

    monkeypatch.setenv("TS_AGENTS_DAYTONA_INSTALL_EXTRAS", "recommended")
    bare = ExecutionContext(sandbox_mode=SandboxMode.DAYTONA)
    assert _context_with_workflow_extras(bare, workflow, kwargs, SandboxMode.DAYTONA) is bare


def test_run_serialized_workflow_bundles_remote_artifacts(monkeypatch, tmp_path):
    import ts_agents.workflows.executor as workflow_executor_mod

    def fake_runner(series_input, *, output_dir, **_kwargs):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path = output_path / "summary.json"
        summary_path.write_text('{"ok": true}')
        return {
            "kind": "workflow",
            "summary": "ok",
            "status": "ok",
            "data": {"output_dir": str(output_path.resolve())},
            "artifacts": [
                {
                    "kind": "json",
                    "path": str(summary_path.resolve()),
                    "mime_type": "application/json",
                    "description": "summary",
                    "created_by": "inspect-series",
                }
            ],
        }

    monkeypatch.setattr(
        workflow_executor_mod,
        "get_workflow",
        lambda name: SimpleNamespace(runner=fake_runner),
    )

    payload = workflow_executor_mod._run_serialized_workflow(
        workflow_name="inspect-series",
        workflow_input=workflow_executor_mod._serialize_workflow_input(
            SeriesInput(
                series=np.array([1.0, 2.0, 3.0]),
                source_type="inline_json",
                label="series",
            )
        ),
        runner_kwargs={"output_dir": "ignored"},
        use_sandbox_artifact_dir=True,
        sandbox_artifact_dir=str(tmp_path / "remote_artifacts"),
        bundle_sandbox_artifacts=True,
    )

    staged_files = payload[workflow_executor_mod._STAGED_WORKFLOW_ARTIFACTS_KEY]
    assert len(staged_files) == 1
    assert staged_files[0]["relative_path"] == "summary.json"
    assert (
        base64.b64decode(staged_files[0]["content_base64"].encode("ascii")).decode("utf-8")
        == '{"ok": true}'
    )


def test_run_serialized_workflow_warns_when_artifact_bundle_limits_are_exceeded(monkeypatch, tmp_path):
    import ts_agents.workflows.executor as workflow_executor_mod

    def fake_runner(series_input, *, output_dir, **_kwargs):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "summary.json").write_text("123456")
        return {
            "kind": "workflow",
            "summary": "ok",
            "status": "ok",
            "data": {"output_dir": str(output_path.resolve())},
            "artifacts": [],
        }

    monkeypatch.setattr(
        workflow_executor_mod,
        "get_workflow",
        lambda name: SimpleNamespace(runner=fake_runner),
    )
    monkeypatch.setenv(workflow_executor_mod._WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV, "4")

    payload = workflow_executor_mod._run_serialized_workflow(
        workflow_name="inspect-series",
        workflow_input=workflow_executor_mod._serialize_workflow_input(
            SeriesInput(
                series=np.array([1.0, 2.0, 3.0]),
                source_type="inline_json",
                label="series",
            )
        ),
        runner_kwargs={"output_dir": "ignored"},
        use_sandbox_artifact_dir=True,
        sandbox_artifact_dir=str(tmp_path / "remote_artifacts"),
        bundle_sandbox_artifacts=True,
    )

    assert workflow_executor_mod._STAGED_WORKFLOW_ARTIFACTS_KEY not in payload
    assert "Skipped remote artifact staging" in payload["warnings"][0]


def test_shared_artifact_bundler_rejects_file_symlinks(tmp_path):
    from ts_agents.tools.artifact_staging import collect_staged_artifact_files

    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("must not be bundled", encoding="utf-8")
    linked = output_dir / "linked-secret.txt"
    try:
        linked.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    payload = {}
    staged = collect_staged_artifact_files(
        output_dir,
        payload,
        max_file_bytes=None,
        max_total_bytes=None,
        file_limit_env="TEST_FILE_LIMIT",
        total_limit_env="TEST_TOTAL_LIMIT",
    )

    assert staged == []
    assert "symlink" in payload["warnings"][0]


def test_workflow_executor_materializes_remote_artifacts_to_requested_output_dir(monkeypatch, tmp_path):
    import ts_agents.workflows.executor as workflow_executor_mod
    from ts_agents.tools.executor import ExecutionContext, ExecutionResult, ExecutionStatus, SandboxMode

    backend_calls = {}
    output_dir = tmp_path / "inspect_daytona"
    remote_output_dir = "/remote/.ts_agents_io/artifacts/inspect-series"

    class FakeDaytonaBackend:
        def is_available(self):
            return True

        def execute(self, tool_name, func, params, context):
            backend_calls["tool_name"] = tool_name
            backend_calls["params"] = params
            backend_calls["context"] = context
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                result={
                    "kind": "workflow",
                    "summary": "ok",
                    "status": "ok",
                    "data": {"output_dir": remote_output_dir},
                    "artifacts": [
                        {
                            "kind": "json",
                            "path": f"{remote_output_dir}/summary.json",
                            "mime_type": "application/json",
                            "description": "summary",
                            "created_by": "inspect-series",
                        },
                        {
                            "kind": "markdown",
                            "path": f"{remote_output_dir}/report.md",
                            "mime_type": "text/markdown",
                            "description": "report",
                            "created_by": "inspect-series",
                        },
                    ],
                    workflow_executor_mod._STAGED_WORKFLOW_ARTIFACTS_KEY: [
                        {
                            "source_path": f"{remote_output_dir}/summary.json",
                            "relative_path": "summary.json",
                            "content_base64": base64.b64encode(b'{"ok": true}').decode("ascii"),
                        },
                        {
                            "source_path": f"{remote_output_dir}/report.md",
                            "relative_path": "report.md",
                            "content_base64": base64.b64encode(b"# report").decode("ascii"),
                        },
                    ],
                },
                formatted_output="ok",
                metadata={"backend": "daytona"},
            )

    executor = workflow_executor_mod.WorkflowExecutor()
    executor.backends[SandboxMode.DAYTONA] = FakeDaytonaBackend()

    monkeypatch.setattr(
        workflow_executor_mod,
        "get_workflow",
        lambda name: SimpleNamespace(
            availability=lambda: {
                "status": "available",
                "available": True,
                "missing_dependencies": [],
                "required_extras": [],
                "optional_features": [],
                "install_hint": None,
            }
        ),
    )
    monkeypatch.setattr(
        workflow_executor_mod,
        "describe_sandbox_backend",
        lambda mode, context=None, backend=None: {
            "backend": mode.value,
            "available": True,
            "reason": None,
            "suggested_fix": None,
            "requirements": [],
            "details": {},
            "description": "backend",
        },
    )

    result = executor.execute(
        "inspect-series",
        SeriesInput(series=np.array([1.0, 2.0, 3.0]), source_type="inline_json", label="series"),
        {"output_dir": str(output_dir), "skip_plots": True},
        context=ExecutionContext(sandbox_mode="daytona"),
    )

    assert result.success is True
    assert backend_calls["tool_name"] == "workflow:inspect-series"
    assert backend_calls["params"]["use_sandbox_artifact_dir"] is True
    assert backend_calls["params"]["bundle_sandbox_artifacts"] is True
    assert backend_calls["params"]["sandbox_artifact_dir"].startswith(".ts_agents_io/artifacts/")
    assert workflow_executor_mod._STAGED_WORKFLOW_ARTIFACTS_KEY not in result.result
    assert result.result["data"]["output_dir"] == str(output_dir.resolve())
    assert (output_dir / "summary.json").read_text() == '{"ok": true}'
    assert (output_dir / "report.md").read_text() == "# report"
    artifact_paths = {artifact["path"] for artifact in result.result["artifacts"]}
    assert str((output_dir / "summary.json").resolve()) in artifact_paths
    assert str((output_dir / "report.md").resolve()) in artifact_paths


def test_materialize_remote_artifacts_rejects_paths_outside_output_dir(tmp_path):
    import ts_agents.workflows.executor as workflow_executor_mod
    from ts_agents.tools.executor import ExecutionResult, ExecutionStatus

    requested_output_dir = tmp_path / "inspect_remote"
    escaped_path = tmp_path / "escaped.txt"
    result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        result={
            "kind": "workflow",
            "summary": "ok",
            "status": "ok",
            "data": {"output_dir": "/remote/output"},
            "artifacts": [
                {
                    "kind": "json",
                    "path": "/remote/output/summary.json",
                    "mime_type": "application/json",
                }
            ],
            workflow_executor_mod._STAGED_WORKFLOW_ARTIFACTS_KEY: [
                {
                    "source_path": "/remote/output/summary.json",
                    "relative_path": "../../escaped.txt",
                    "content_base64": base64.b64encode(b"oops").decode("ascii"),
                }
            ],
        },
        formatted_output="ok",
    )

    workflow_executor_mod._materialize_remote_workflow_output_paths(
        result,
        str(requested_output_dir),
    )

    assert not escaped_path.exists()
    assert not requested_output_dir.exists()
    assert result.result["data"]["output_dir"] == "/remote/output"
    assert result.result["artifacts"][0]["path"] == "/remote/output/summary.json"
    assert any("escapes the output directory" in warning for warning in result.result["warnings"])
    assert any("remains inaccessible on the host" in warning for warning in result.result["warnings"])


def test_materialize_remote_artifacts_warns_when_artifact_path_was_not_staged(tmp_path):
    import ts_agents.workflows.executor as workflow_executor_mod
    from ts_agents.tools.executor import ExecutionResult, ExecutionStatus

    output_dir = tmp_path / "inspect_remote"
    remote_output_dir = "/remote/output"
    result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        result={
            "kind": "workflow",
            "summary": "ok",
            "status": "ok",
            "data": {"output_dir": remote_output_dir},
            "artifacts": [
                {
                    "kind": "json",
                    "path": f"{remote_output_dir}/summary.json",
                    "mime_type": "application/json",
                },
                {
                    "kind": "markdown",
                    "path": f"{remote_output_dir}/report.md",
                    "mime_type": "text/markdown",
                },
            ],
            workflow_executor_mod._STAGED_WORKFLOW_ARTIFACTS_KEY: [
                {
                    "source_path": f"{remote_output_dir}/summary.json",
                    "relative_path": "summary.json",
                    "content_base64": base64.b64encode(b'{"ok": true}').decode("ascii"),
                }
            ],
        },
        formatted_output="ok",
    )

    workflow_executor_mod._materialize_remote_workflow_output_paths(result, str(output_dir))

    assert result.result["data"]["output_dir"] == str(output_dir.resolve())
    assert result.result["artifacts"][0]["path"] == str((output_dir / "summary.json").resolve())
    assert result.result["artifacts"][1]["path"] == f"{remote_output_dir}/report.md"
    assert any(
        "report.md' was not staged and remains inaccessible on the host" in warning
        for warning in result.result["warnings"]
    )


def test_materialize_remote_artifacts_ignores_empty_staged_bundles(tmp_path):
    import ts_agents.workflows.executor as workflow_executor_mod
    from ts_agents.tools.executor import ExecutionResult, ExecutionStatus

    output_dir = tmp_path / "inspect_remote"
    result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        result={
            "kind": "workflow",
            "summary": "ok",
            "status": "ok",
            "data": {"output_dir": "/remote/output"},
            "artifacts": [],
            workflow_executor_mod._STAGED_WORKFLOW_ARTIFACTS_KEY: [],
        },
        formatted_output="ok",
    )

    workflow_executor_mod._materialize_remote_workflow_output_paths(result, str(output_dir))

    assert not output_dir.exists()
    assert result.result["data"]["output_dir"] == "/remote/output"


def test_activity_workflow_availability_is_degraded_without_aeon(monkeypatch):
    import ts_agents.workflows as workflows_mod

    monkeypatch.setattr(
        workflows_mod,
        "_module_available",
        lambda module_name: module_name in {"sklearn", "matplotlib"},
    )

    availability = workflows_mod.get_workflow("activity-recognition").availability()

    assert availability["available"] is True
    assert availability["status"] == "degraded"
    rocket_feature = next(
        feature for feature in availability["optional_features"] if feature["name"] == "rocket_backends"
    )
    assert rocket_feature["available"] is False


def test_workflow_run_forecast_series_writes_expected_files(monkeypatch, capsys, tmp_path):
    import ts_agents.core.comparison as comparison_mod
    import ts_agents.workflows.forecast as forecast_workflow

    csv_path = tmp_path / "series.csv"
    csv_path.write_text("ds,y\n2024-01-01,1.0\n2024-01-02,1.1\n2024-01-03,1.2\n2024-01-04,1.3\n")

    def fake_compare_forecasting_methods(
        series,
        horizon,
        methods,
        validation_size=None,
        season_length=None,
    ):
        return ComparisonResult(
            category="forecasting",
            methods=list(methods),
            results={},
            metrics={
                "arima": {"mae": 0.3, "rmse": 0.3, "mape": 10.0},
                "theta": {"mae": 0.1, "rmse": 0.1, "mape": 4.0},
            },
            rankings={"rmse": ["theta", "arima"], "mae": ["theta", "arima"], "mape": ["theta", "arima"]},
            recommendation="theta is the best baseline.",
            computation_times={"arima": 0.01, "theta": 0.01},
        )

    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)
    monkeypatch.setattr(
        forecast_workflow,
        "_forecast_with_method",
        lambda series, method, horizon, season_length=None: SimpleNamespace(
            forecast=np.array([1.4, 1.5])
        ),
    )

    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input",
            str(csv_path),
            "--time-col",
            "ds",
            "--value-col",
            "y",
            "--horizon",
            "2",
            "--methods",
            "arima,theta",
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
    assert payload["requires_review"] is False
    assert payload["name"] == "forecast-series"
    assert payload["result"]["data"]["best_method"] == "theta"
    assert (output_dir / "forecast_comparison.json").exists()
    assert (output_dir / "forecast.json").exists()
    assert (output_dir / "forecast.csv").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "run_manifest.json").exists()
    artifact_paths = {artifact["path"] for artifact in payload["result"]["artifacts"]}
    assert str(output_dir / "forecast_comparison.json") in artifact_paths
    assert str(output_dir / "forecast.json") in artifact_paths
    assert str(output_dir / "forecast.csv") in artifact_paths
    assert str(output_dir / "report.md") in artifact_paths
    assert str(output_dir / "run_manifest.json") in artifact_paths


def _fake_series_foundation(monkeypatch):
    from ts_agents.core.forecasting import foundation

    calls = []
    cleared = []

    def fake_forecast_arrays(arrays, *, model, horizon, context_length=None, accelerator="cpu", seed=0):
        calls.append(dict(model=model, horizon=horizon, context_length=context_length, accelerator=accelerator))
        return [np.full(horizon, float(np.asarray(arrays[0])[-1])) for _ in arrays]

    monkeypatch.setattr(foundation, "forecast_arrays", fake_forecast_arrays)
    monkeypatch.setattr(foundation, "clear_model_cache", lambda: cleared.append(True))
    return calls, cleared


def test_workflow_run_forecast_series_mocked_foundation_model(monkeypatch, capsys, tmp_path):
    calls, cleared = _fake_series_foundation(monkeypatch)
    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input-json",
            json.dumps({"series": [float(i % 4) for i in range(24)]}),
            "--horizon",
            "4",
            "--methods",
            "seasonal_naive,chronos2_small",
            "--season-length",
            "4",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    data = payload["result"]["data"]
    assert data["valid_methods"] == ["seasonal_naive", "chronos2_small"]
    assert data["best_method"] == "seasonal_naive"
    assert data["foundation_models"]["chronos2_small"]["hub_model_name"] == "autogluon/chronos-2-small"
    assert data["foundation_models"]["chronos2_small"]["input_chunk_length"] == [1, 512]
    assert calls == [dict(model="chronos2_small", horizon=4, context_length=None, accelerator="cpu")]
    assert cleared == [True]
    report = (output_dir / "report.md").read_text()
    assert "zero-shot" in report
    assert "autogluon/chronos-2-small@" in report and "licence" in report
    # Provenance is persisted, not only returned in the envelope.
    spec = json.loads((output_dir / "models" / "chronos2_small" / "model_spec.json").read_text())
    assert spec == data["foundation_models"]["chronos2_small"]
    assert spec["weights_saved"] is False and spec["license"]
    artifact_paths = {artifact["path"] for artifact in payload["result"]["artifacts"]}
    assert str(output_dir / "models" / "chronos2_small" / "model_spec.json") in artifact_paths
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert "context_length" not in manifest["resume_identity"]["options"]


def _run_forecast_series_cli(tmp_path, methods, *extra_args):
    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input-json",
            json.dumps({"series": [float(i % 4) for i in range(24)]}),
            "--horizon",
            "4",
            "--methods",
            methods,
            "--season-length",
            "4",
            *extra_args,
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    return code, output_dir


def test_forecast_series_failed_foundation_model_is_not_reported_as_run(monkeypatch, capsys, tmp_path):
    from ts_agents.core.forecasting import foundation

    def unavailable(arrays, **kwargs):
        raise foundation.FoundationModelUnavailableError(
            "weights for autogluon/chronos-2@abc are not cached and could not be downloaded"
        )

    monkeypatch.setattr(foundation, "forecast_arrays", unavailable)
    code, output_dir = _run_forecast_series_cli(tmp_path, "seasonal_naive,chronos2")

    assert code == 0
    data = json.loads(capsys.readouterr().out)["result"]["data"]
    assert data["failed_methods"] == ["chronos2"]
    assert "foundation_models" not in data
    report = (output_dir / "report.md").read_text()
    assert "ran zero-shot" not in report
    assert "Requested but failed" in report and "chronos2" in report
    assert not (output_dir / "models").exists()


def test_forecast_series_all_weights_unavailable_is_typed_backend_unavailable(
    monkeypatch, capsys, tmp_path
):
    from ts_agents.core.forecasting import foundation

    def unavailable(arrays, **kwargs):
        raise foundation.FoundationModelUnavailableError(
            "weights for autogluon/chronos-2-small@abc are not cached and could not be "
            "downloaded; pre-populate HF_HOME"
        )

    monkeypatch.setattr(foundation, "forecast_arrays", unavailable)
    code, _ = _run_forecast_series_cli(tmp_path, "chronos2_small")

    assert code == 5
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "backend_unavailable"
    assert error["retryable"] is True
    assert "HF_HOME" in error["hint"]
    assert error["details"]["exception_type"] == "FoundationModelUnavailableError"


def test_forecast_series_rejects_invalid_context_length_without_foundation_models(capsys, tmp_path):
    code, output_dir = _run_forecast_series_cli(
        tmp_path, "seasonal_naive", "--context-length", "0"
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "validation_error"
    assert "context_length" in payload["error"]["message"]


def test_forecast_series_foundation_options_are_forwarded(monkeypatch, tmp_path):
    from ts_agents.workflows.forecast import run_forecast_series_workflow

    calls, _ = _fake_series_foundation(monkeypatch)
    series = SeriesInput(series=np.arange(40, dtype=float), source_type="inline_json", label="ramp")
    payload = run_forecast_series_workflow(
        series,
        output_dir=str(tmp_path / "run"),
        horizon=6,
        validation_size=4,
        methods=["chronos2_small"],
        context_length=16,
        accelerator="cpu",
        skip_plots=True,
    )

    # Holdout validation at 4 steps, then the final forecast at 6 steps.
    assert [(call["horizon"], call["context_length"]) for call in calls] == [(4, 16), (6, 16)]
    assert payload.data["best_method"] == "chronos2_small"
    assert len(payload.data["forecast"]) == 6

    spec = payload.data["foundation_models"]["chronos2_small"]
    assert spec["forecast_phase"] == "future"
    assert spec["output_chunk_length"] == 6
    assert spec["validation_spec"]["output_chunk_length"] == 6
    assert spec["validation_spec"]["forecast_horizon"] == 4
    assert spec["input_chunk_length"] == [1, 16]


def test_forecast_series_nonwinning_foundation_spec_records_validation_horizon(monkeypatch, tmp_path):
    from ts_agents.workflows.forecast import run_forecast_series_workflow

    calls, _ = _fake_series_foundation(monkeypatch)
    series = SeriesInput(
        series=np.tile(np.arange(4.0), 10), source_type="inline_json", label="seasonal"
    )
    payload = run_forecast_series_workflow(
        series, output_dir=str(tmp_path / "run"), horizon=3, validation_size=4,
        methods=["seasonal_naive", "chronos2_small"], season_length=4, skip_plots=True,
    )
    assert payload.data["best_method"] == "seasonal_naive"
    assert [call["horizon"] for call in calls] == [4]
    spec = payload.data["foundation_models"]["chronos2_small"]
    assert spec["forecast_phase"] == "validation"
    assert spec["output_chunk_length"] == 4
    assert "validation_spec" not in spec
    persisted = json.loads((tmp_path / "run/models/chronos2_small/model_spec.json").read_text())
    assert persisted == spec


@pytest.mark.parametrize(
    ("extra_args", "expected_code"),
    [(["--horizon", "2000"], 2), (["--horizon", "4"], 3)],
)
def test_workflow_run_forecast_series_foundation_error_paths(
    monkeypatch, capsys, tmp_path, extra_args, expected_code
):
    from ts_agents.core.forecasting import catalog

    monkeypatch.setattr(
        catalog, "missing_modules", lambda name: list(catalog.get_method(name).modules)
    )
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input-json",
            json.dumps({"series": [float(i) for i in range(30)]}),
            *extra_args,
            "--methods",
            "chronos2_small",
            "--skip-plots",
            "--output-dir",
            str(tmp_path / "forecast"),
            "--json",
        ]
    )

    assert code == expected_code
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    if expected_code == 3:
        assert "ts-agents[foundation]" in payload["error"]["message"]
    else:
        assert "horizon" in payload["error"]["message"]


def test_workflow_run_forecast_series_reports_degraded_when_some_methods_fail(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.comparison as comparison_mod
    import ts_agents.workflows.forecast as forecast_workflow

    csv_path = tmp_path / "series.csv"
    csv_path.write_text("ds,y\n2024-01-01,1.0\n2024-01-02,1.1\n2024-01-03,1.2\n2024-01-04,1.3\n")

    def fake_compare_forecasting_methods(
        series,
        horizon,
        methods,
        validation_size=None,
        season_length=None,
    ):
        return ComparisonResult(
            category="forecasting",
            methods=["theta"],
            results={},
            metrics={
                "arima": {
                    "error": 'Statistical forecasting requires optional dependencies. Install with: pip install "ts-agents[forecasting]"'
                },
                "theta": {"mae": 0.1, "rmse": 0.1, "mape": 4.0},
            },
            rankings={"rmse": ["theta"], "mae": ["theta"], "mape": ["theta"]},
            recommendation="theta is the best baseline.",
            computation_times={"theta": 0.01},
        )

    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)
    monkeypatch.setattr(
        forecast_workflow,
        "_forecast_with_method",
        lambda series, method, horizon, season_length=None: SimpleNamespace(
            forecast=np.array([1.4, 1.5])
        ),
    )

    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input",
            str(csv_path),
            "--time-col",
            "ds",
            "--value-col",
            "y",
            "--horizon",
            "2",
            "--methods",
            "arima,theta",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["quality_status"] == "degraded"
    assert payload["degraded"] is True
    assert payload["requires_review"] is True
    assert payload["result"]["status"] == "degraded"
    assert payload["result"]["data"]["valid_methods"] == ["theta"]
    assert payload["result"]["data"]["failed_methods"] == ["arima"]
    assert "partial_method_failure" in payload["result"]["data"]["quality_flags"]
    assert "only_one_valid_method" in payload["result"]["data"]["quality_flags"]
    assert payload["result"]["data"]["best_method"] == "theta"


def test_workflow_run_forecast_series_treats_missing_metrics_as_failed_method(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.comparison as comparison_mod
    import ts_agents.workflows.forecast as forecast_workflow

    csv_path = tmp_path / "series.csv"
    csv_path.write_text("ds,y\n2024-01-01,1.0\n2024-01-02,1.1\n2024-01-03,1.2\n2024-01-04,1.3\n")

    def fake_compare_forecasting_methods(
        series,
        horizon,
        methods,
        validation_size=None,
        season_length=None,
    ):
        return ComparisonResult(
            category="forecasting",
            methods=["theta"],
            results={},
            metrics={
                "theta": {"mae": 0.1, "rmse": 0.1, "mape": 4.0},
            },
            rankings={"rmse": ["theta"], "mae": ["theta"], "mape": ["theta"]},
            recommendation="theta is the best baseline.",
            computation_times={"theta": 0.01},
        )

    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)
    monkeypatch.setattr(
        forecast_workflow,
        "_forecast_with_method",
        lambda series, method, horizon, season_length=None: SimpleNamespace(
            forecast=np.array([1.4, 1.5])
        ),
    )

    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input",
            str(csv_path),
            "--time-col",
            "ds",
            "--value-col",
            "y",
            "--horizon",
            "2",
            "--methods",
            "arima,theta",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["status"] == "degraded"
    assert payload["result"]["data"]["valid_methods"] == ["theta"]
    assert payload["result"]["data"]["failed_methods"] == ["arima"]
    assert "missing_method_metrics" in payload["result"]["data"]["quality_flags"]


def test_workflow_run_forecast_series_fails_when_all_methods_fail(monkeypatch, capsys, tmp_path):
    import ts_agents.core.comparison as comparison_mod

    csv_path = tmp_path / "series.csv"
    csv_path.write_text("ds,y\n2024-01-01,1.0\n2024-01-02,1.1\n2024-01-03,1.2\n2024-01-04,1.3\n")

    def fake_compare_forecasting_methods(
        series,
        horizon,
        methods,
        validation_size=None,
        season_length=None,
    ):
        return ComparisonResult(
            category="forecasting",
            methods=[],
            results={},
            metrics={
                "arima": {
                    "error": 'Statistical forecasting requires optional dependencies. Install with: pip install "ts-agents[forecasting]"'
                },
                "theta": {
                    "error": 'Statistical forecasting requires optional dependencies. Install with: pip install "ts-agents[forecasting]"'
                },
            },
            rankings={},
            recommendation="Unable to generate recommendation - no valid results.",
            computation_times={},
        )

    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)

    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input",
            str(csv_path),
            "--time-col",
            "ds",
            "--value-col",
            "y",
            "--horizon",
            "2",
            "--methods",
            "arima,theta",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "dependency_error"
    assert "All forecast methods failed" in payload["error"]["message"]
    assert not (output_dir / "forecast.json").exists()


def test_workflow_run_forecast_series_classifies_importerror_by_error_type(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.comparison as comparison_mod

    csv_path = tmp_path / "series.csv"
    csv_path.write_text("ds,y\n2024-01-01,1.0\n2024-01-02,1.1\n2024-01-03,1.2\n2024-01-04,1.3\n")

    def fake_compare_forecasting_methods(
        series,
        horizon,
        methods,
        validation_size=None,
        season_length=None,
    ):
        return ComparisonResult(
            category="forecasting",
            methods=[],
            results={},
            metrics={
                "arima": {
                    "error": "backend package unavailable",
                    "error_type": "ImportError",
                },
                "theta": {
                    "error": "backend package unavailable",
                    "error_type": "ModuleNotFoundError",
                },
            },
            rankings={},
            recommendation="Unable to generate recommendation - no valid results.",
            computation_times={},
        )

    monkeypatch.setattr(comparison_mod, "compare_forecasting_methods", fake_compare_forecasting_methods)

    output_dir = tmp_path / "forecast"
    code = run(
        [
            "workflow",
            "run",
            "forecast-series",
            "--input",
            str(csv_path),
            "--time-col",
            "ds",
            "--value-col",
            "y",
            "--horizon",
            "2",
            "--methods",
            "arima,theta",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "dependency_error"


def test_workflow_run_activity_recognition_writes_expected_files(monkeypatch, capsys, tmp_path):
    import ts_agents.core.windowing as windowing

    class FakeSelection:
        best_window_size = 96
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 96,
                "metric": "balanced_accuracy",
                "scores_by_window": {32: 0.66, 96: 0.84},
                "n_windows_by_window": {32: 180, 96: 120},
                "details": {},
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 0.84

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 96,
                "stride": 48,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 0.84,
                "n_windows": 120,
                "class_counts": {"idle": 62, "walk": 58},
                "classification": {
                    "method": "minirocket",
                    "accuracy": 0.86,
                    "f1_score": 0.83,
                    "confusion_matrix": [[26, 4], [5, 25]],
                    "predictions": ["idle", "walk"],
                },
            }

    monkeypatch.setattr(windowing, "select_window_size", lambda *args, **kwargs: FakeSelection())
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", lambda *args, **kwargs: FakeEval())

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )

    output_dir = tmp_path / "activity"
    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--window-sizes",
            "32,96",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "activity-recognition"
    assert payload["result"]["data"]["best_window_size"] == 96
    assert payload["result"]["data"]["classifier_requested"] == "auto"
    assert payload["result"]["data"]["classifier_resolved"] == "minirocket"
    assert payload["result"]["data"]["classifier_effective_backend"] == "minirocket"
    assert payload["result"]["data"]["classification_method"] == "minirocket"
    assert payload["result"]["data"]["classifier_used"] == "minirocket"
    assert (output_dir / "window_selection.json").exists()
    assert (output_dir / "eval.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "run_manifest.json").exists()
    artifact_paths = {artifact["path"] for artifact in payload["result"]["artifacts"]}
    assert str(output_dir / "window_selection.json") in artifact_paths
    assert str(output_dir / "eval.json") in artifact_paths
    assert str(output_dir / "report.md") in artifact_paths
    assert str(output_dir / "run_manifest.json") in artifact_paths


def test_workflow_run_activity_recognition_surfaces_effective_backend_provenance(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.windowing as windowing

    monkeypatch.chdir(tmp_path)

    class FakeSelection:
        best_window_size = 96
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 96,
                "metric": "balanced_accuracy",
                "scores_by_window": {32: 0.66, 96: 0.84},
                "n_windows_by_window": {32: 180, 96: 120},
                "details": {},
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 0.84

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 96,
                "stride": 48,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 0.84,
                "n_windows": 120,
                "class_counts": {"idle": 62, "walk": 58},
                "classification": {
                    "method": "rocket_fallback",
                    "accuracy": 0.86,
                    "f1_score": 0.83,
                    "confusion_matrix": [[26, 4], [5, 25]],
                    "predictions": ["idle", "walk"],
                },
            }

    monkeypatch.setattr(windowing, "select_window_size", lambda *args, **kwargs: FakeSelection())
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", lambda *args, **kwargs: FakeEval())

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )

    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--window-sizes",
            "32,96",
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    data = payload["result"]["data"]
    assert data["classifier_requested"] == "auto"
    assert data["classifier_resolved"] == "minirocket"
    assert data["classifier_evaluation"] == "minirocket"
    assert data["classification_method"] == "rocket_fallback"
    assert data["classifier_effective_backend"] == "rocket_fallback"
    assert data["classifier_used"] == "rocket_fallback"


def test_workflow_run_activity_recognition_sanitizes_nan_scores_and_surfaces_quality_flags(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.windowing as windowing

    class FakeSelection:
        best_window_size = 96
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 96,
                "metric": "balanced_accuracy",
                "scores_by_window": {32: float("nan"), 96: 1.0},
                "n_windows_by_window": {32: 7, 96: 12},
                "details": {},
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 1.0

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 96,
                "stride": 48,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 1.0,
                "n_windows": 12,
                "class_counts": {"idle": 8, "walk": 4, "jog": 3},
                "classification": {
                    "method": "minirocket",
                    "accuracy": 1.0,
                    "f1_score": 1.0,
                    "confusion_matrix": [[8, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "predictions": ["idle"],
                },
            }

    monkeypatch.setattr(windowing, "select_window_size", lambda *args, **kwargs: FakeSelection())
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", lambda *args, **kwargs: FakeEval())

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )

    output_dir = tmp_path / "activity"
    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--window-sizes",
            "32,96",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    stdout = capsys.readouterr().out
    assert "NaN" not in stdout
    payload = json.loads(
        stdout,
        parse_constant=lambda constant: (_ for _ in ()).throw(AssertionError(constant)),
    )
    assert payload["quality_status"] == "degraded"
    assert payload["degraded"] is True
    assert payload["requires_review"] is True
    assert payload["result"]["status"] == "degraded"
    flags = set(payload["result"]["data"]["quality_flags"])
    assert "nan_window_scores" in flags
    assert "too_few_windows" in flags
    assert "single_class_test_split" in flags
    assert "metric_not_comparable" in flags
    assert "perfect_metrics" in flags
    assert payload["result"]["data"]["window_selection"]["scores_by_window"]["32"] is None


def test_activity_window_selection_plot_ignores_null_scores():
    import ts_agents.workflows.activity as activity_workflow

    fig = activity_workflow._plot_window_selection(
        {
            "best_window_size": 96,
            "metric": "balanced_accuracy",
            "scores_by_window": {"32": None, "96": 0.84},
        }
    )

    assert fig is not None
    activity_workflow._close_plots(fig)


def test_workflow_run_activity_recognition_continues_when_plot_generation_fails(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.windowing as windowing
    import ts_agents.workflows.activity as activity_workflow

    class FakeSelection:
        best_window_size = 96
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 96,
                "metric": "balanced_accuracy",
                "scores_by_window": {32: 0.66, 96: 0.84},
                "n_windows_by_window": {32: 180, 96: 120},
                "details": {},
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 0.84

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 96,
                "stride": 48,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 0.84,
                "n_windows": 120,
                "class_counts": {"idle": 62, "walk": 58},
                "classification": {
                    "method": "minirocket",
                    "accuracy": 0.86,
                    "f1_score": 0.83,
                    "confusion_matrix": [[26, 4], [5, 25]],
                    "predictions": ["idle", "walk"],
                },
            }

    def raise_plot_error(_payload):
        raise ValueError("plot payload incomplete")

    monkeypatch.setattr(windowing, "select_window_size", lambda *args, **kwargs: FakeSelection())
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", lambda *args, **kwargs: FakeEval())
    monkeypatch.setattr(activity_workflow, "_plot_window_selection", raise_plot_error)

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )

    output_dir = tmp_path / "activity"
    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["result"]["warnings"] == ["Skipping activity-recognition plots: plot payload incomplete"]
    assert (output_dir / "window_selection.json").exists()
    assert (output_dir / "eval.json").exists()
    assert (output_dir / "report.md").exists()
    assert not (output_dir / "window_scores.png").exists()
    assert not (output_dir / "confusion_matrix.png").exists()


def test_workflow_run_activity_recognition_threads_new_windowing_controls(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.windowing as windowing

    observed = {}

    class FakeSelection:
        best_window_size = 64
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 64,
                "metric": "balanced_accuracy",
                "scores_by_window": {64: 0.78},
                "n_windows_by_window": {64: 24},
                "details": {
                    "n_splits": 5,
                    "window_diagnostics": {
                        64: {
                            "n_windows": 24,
                            "retained_classes": ["idle", "walk"],
                            "dropped_classes": [],
                            "valid_split_count": 5,
                        }
                    },
                },
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 0.76

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 64,
                "stride": 16,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 0.76,
                "n_windows": 24,
                "n_splits": 5,
                "split_scores": [0.74, 0.78],
                "retained_classes": ["idle", "walk"],
                "dropped_classes": [],
                "class_counts": {"idle": 12, "walk": 12},
                "details": {"valid_split_count": 5},
                "classification": {
                    "method": "minirocket",
                    "accuracy": 0.79,
                    "f1_score": 0.75,
                    "confusion_matrix": [[6, 2], [3, 5]],
                    "predictions": ["idle", "walk"],
                },
            }

    def fake_select_window_size(*args, **kwargs):
        observed["selection_kwargs"] = kwargs
        return FakeSelection()

    def fake_evaluate_windowed_classifier(*args, **kwargs):
        observed["evaluation_kwargs"] = kwargs
        return FakeEval()

    monkeypatch.setattr(windowing, "select_window_size", fake_select_window_size)
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", fake_evaluate_windowed_classifier)

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )

    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--labeling",
            "majority",
            "--stride",
            "16",
            "--n-splits",
            "5",
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert observed["selection_kwargs"]["labeling"] == "majority"
    assert observed["selection_kwargs"]["stride"] == 16
    assert observed["selection_kwargs"]["n_splits"] == 5
    assert observed["evaluation_kwargs"]["labeling"] == "majority"
    assert observed["evaluation_kwargs"]["stride"] == 16
    assert observed["evaluation_kwargs"]["n_splits"] == 5
    assert payload["result"]["data"]["labeling"] == "majority"
    assert payload["result"]["data"]["stride_requested"] == 16
    assert payload["result"]["data"]["n_splits"] == 5


def test_workflow_run_activity_recognition_report_surfaces_retention_drops(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.windowing as windowing

    class FakeSelection:
        best_window_size = 96
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 96,
                "metric": "balanced_accuracy",
                "scores_by_window": {32: 0.72, 96: 0.91},
                "n_windows_by_window": {32: 40, 96: 12},
                "details": {
                    "n_splits": 3,
                    "window_diagnostics": {
                        32: {
                            "n_windows": 40,
                            "retained_classes": ["idle", "walk", "stairs"],
                            "dropped_classes": [],
                            "valid_split_count": 3,
                        },
                        96: {
                            "n_windows": 12,
                            "retained_classes": ["idle", "walk"],
                            "dropped_classes": ["stairs"],
                            "valid_split_count": 1,
                        },
                    },
                },
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 0.91

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 96,
                "stride": 48,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 0.91,
                "n_windows": 12,
                "n_splits": 1,
                "split_scores": [0.91],
                "retained_classes": ["idle", "walk"],
                "dropped_classes": ["stairs"],
                "class_counts": {"idle": 7, "walk": 5},
                "details": {"valid_split_count": 1},
                "classification": {
                    "method": "minirocket",
                    "accuracy": 0.93,
                    "f1_score": 0.9,
                    "confusion_matrix": [[4, 1], [0, 5]],
                    "predictions": ["idle", "walk"],
                },
            }

    monkeypatch.setattr(windowing, "select_window_size", lambda *args, **kwargs: FakeSelection())
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", lambda *args, **kwargs: FakeEval())

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )
    output_dir = tmp_path / "activity"

    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--skip-plots",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality_status"] == "degraded"
    report_text = (output_dir / "report.md").read_text()
    assert "Window Sweep" in report_text
    assert "Dropped Classes" in report_text
    assert "`stairs`" in report_text
    assert "No obvious pathologies" not in report_text


def test_workflow_run_activity_recognition_surfaces_classifier_backend_warnings(
    monkeypatch,
    capsys,
    tmp_path,
):
    import ts_agents.core.windowing as windowing

    class FakeSelection:
        best_window_size = 64
        metric = "balanced_accuracy"

        def to_dict(self):
            return {
                "method": "window_size_selection",
                "best_window_size": 64,
                "metric": "balanced_accuracy",
                "scores_by_window": {64: 0.8},
                "n_windows_by_window": {64: 20},
                "details": {"n_splits": 3, "window_diagnostics": {64: {"n_windows": 20, "retained_classes": ["idle", "walk"], "dropped_classes": [], "valid_split_count": 3}}},
            }

    class FakeEval:
        metric = "balanced_accuracy"
        score = 0.8

        def to_dict(self):
            return {
                "method": "windowed_classification",
                "window_size": 64,
                "stride": 32,
                "classifier": "minirocket",
                "metric": "balanced_accuracy",
                "score": 0.8,
                "n_windows": 20,
                "n_splits": 3,
                "split_scores": [0.8],
                "retained_classes": ["idle", "walk"],
                "dropped_classes": [],
                "class_counts": {"idle": 10, "walk": 10},
                "classification": {
                    "method": "rocket_fallback",
                    "warnings": ["ROCKET backend fallback activated during fit/predict: RuntimeError: broken aeon"],
                    "accuracy": 0.8,
                    "f1_score": 0.8,
                    "confusion_matrix": [[4, 1], [1, 4]],
                    "predictions": ["idle", "walk"],
                },
            }

    monkeypatch.setattr(windowing, "select_window_size", lambda *args, **kwargs: FakeSelection())
    monkeypatch.setattr(windowing, "evaluate_windowed_classifier", lambda *args, **kwargs: FakeEval())

    csv_path = tmp_path / "stream.csv"
    csv_path.write_text(
        "x,y,z,label\n"
        "0,0,0,idle\n"
        "1,1,1,idle\n"
        "2,2,2,walk\n"
        "3,3,3,walk\n"
    )

    code = run(
        [
            "workflow",
            "run",
            "activity-recognition",
            "--input",
            str(csv_path),
            "--label-col",
            "label",
            "--value-cols",
            "x,y,z",
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["degraded"] is True
    assert "classifier_backend_warning" in payload["result"]["data"]["quality_flags"]
    assert any("fallback activated" in warning for warning in payload["result"]["warnings"])


def test_handle_workflow_command_uses_registry_runner(monkeypatch):
    import importlib
    import ts_agents.workflows as workflows
    import ts_agents.workflows.executor as workflow_executor

    cli_main = importlib.import_module("ts_agents.cli.main")
    observed = {}
    fake_series_input = SimpleNamespace(provenance={"series_ref": {"source_type": "inline_json"}})

    def fake_runner(series_input, **kwargs):
        observed["series_input"] = series_input
        observed["kwargs"] = kwargs
        return {"status": "ok", "summary": "fake workflow", "data": {}, "artifacts": []}

    def fake_build_runner_kwargs(args):
        observed["builder_args"] = args.workflow_name
        return {"output_dir": "outputs/custom", "skip_plots": True}

    def fake_load_input(args):
        observed["loader_args"] = args.workflow_name
        return fake_series_input

    monkeypatch.setattr(
        workflows,
        "get_workflow",
        lambda name: SimpleNamespace(
            name=name,
            runner=fake_runner,
            load_input=fake_load_input,
            build_runner_kwargs=fake_build_runner_kwargs,
        ),
    )
    monkeypatch.setattr(
        workflow_executor,
        "execute_workflow",
        lambda workflow_name, workflow_input, runner_kwargs, context=None: SimpleNamespace(
            success=True,
            result=fake_runner(workflow_input, **runner_kwargs),
            formatted_output="",
            metadata={},
        ),
    )

    args = argparse.Namespace(
        workflow_command="run",
        workflow_name="inspect-series",
        input=None,
        input_json='{"series":[1,2,3]}',
        stdin=False,
        run_id=None,
        variable=None,
        time_col=None,
        value_col=None,
        use_test_data=False,
        full_data=False,
        output_dir="outputs/inspect",
        max_lag=None,
        skip_plots=True,
        sandbox=None,
        allow_network=False,
        allow_fallback=False,
        fallback_backend=None,
    )

    result, text = cli_main._handle_workflow_command(args)

    assert text is None
    assert result["status"] == "ok"
    assert observed["series_input"] is fake_series_input
    assert observed["loader_args"] == "inspect-series"
    assert observed["builder_args"] == "inspect-series"
    assert observed["kwargs"]["output_dir"] == "outputs/custom"
    assert observed["kwargs"]["skip_plots"] is True
    assert observed["kwargs"]["run_id"]
    assert observed["kwargs"]["resumed"] is False
    assert observed["kwargs"]["output_dir_mode"] == "generated"
    assert args._ts_input_payload["options"]["output_dir"] == "outputs/custom"
    assert args._ts_input_payload["options"]["skip_plots"] is True
    assert args._ts_input_payload["options"]["run_id"] == observed["kwargs"]["run_id"]
    assert args._ts_input_payload["run"]["run_id"] == observed["kwargs"]["run_id"]


def test_workflow_run_inspect_series_returns_absolute_paths_for_relative_output_dir(
    capsys,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    code = run(
        [
            "workflow",
            "run",
            "inspect-series",
            "--input-json",
            '{"series":[1,2,3,4,5]}',
            "--output-dir",
            "outputs/inspect",
            "--skip-plots",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    expected_output_dir = str((tmp_path / "outputs" / "inspect").resolve())
    assert payload["result"]["data"]["output_dir"] == expected_output_dir
    for artifact in payload["result"]["artifacts"]:
        assert Path(artifact["path"]).is_absolute()
