"""Tests for tool executor serialization and context handling."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ts_agents.contracts import ArtifactRef, ToolPayload
from ts_agents.core.base import PeakResult
from ts_agents.tools.executor import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    LocalBackend,
    SandboxMode,
    ToolErrorCode,
    ToolExecutor,
    _persist_docker_artifacts,
    _persist_subprocess_artifacts,
    describe_sandbox_backend,
)
from ts_agents.tools.results import DecompositionResult as ToolDecompositionResult


def test_execution_context_coerces_string():
    ctx = ExecutionContext(sandbox_mode="docker")
    assert ctx.sandbox_mode == SandboxMode.DOCKER

    ctx = ExecutionContext(sandbox_mode="LOCAL")
    assert ctx.sandbox_mode == SandboxMode.LOCAL

    ctx = ExecutionContext(sandbox_mode="docker", fallback_backend="local")
    assert ctx.fallback_backend == SandboxMode.LOCAL


def test_execution_context_daytona_bootstrap_defaults():
    ctx = ExecutionContext(sandbox_mode="daytona")
    assert ctx.sandbox_mode == SandboxMode.DAYTONA
    assert ctx.daytona_snapshot == "daytonaio/sandbox:0.4.3"
    assert ctx.daytona_repo_url == "https://github.com/fnauman/ts-agents"
    assert ctx.daytona_install_editable is True
    assert ctx.daytona_stream_logs is True
    assert ctx.daytona_log_file is None
    assert ctx.modal_stream_logs is True
    assert ctx.modal_log_file is None


def test_execution_result_serializes_analysis_result():
    result = PeakResult(
        method="test",
        peak_indices=np.array([1, 2]),
        peak_values=np.array([0.1, 0.2]),
        count=2,
    )

    exec_result = ExecutionResult(status=ExecutionStatus.SUCCESS, result=result)
    payload = exec_result.to_dict()

    assert payload["result"]["peak_indices"] == [1, 2]
    assert payload["result"]["peak_values"] == [0.1, 0.2]
    json.dumps(payload)


def test_execution_result_serializes_tool_result():
    result = ToolDecompositionResult(
        trend=[1.0],
        seasonal=[],
        residual=[0.0],
        period=1,
        method="stl",
    )

    exec_result = ExecutionResult(status=ExecutionStatus.SUCCESS, result=result)
    payload = exec_result.to_dict()

    assert payload["result"]["trend"] == [1.0]
    assert payload["result"]["period"] == 1
    json.dumps(payload)


def test_execution_result_serializes_tool_payload():
    result = ToolPayload(
        kind="statistics",
        summary="Computed stats.",
        data={"mean": 1.5},
        artifacts=[
            ArtifactRef(
                kind="image",
                path="/tmp/stats.png",
                mime_type="image/png",
                description="Stats plot",
            )
        ],
    )

    exec_result = ExecutionResult(status=ExecutionStatus.SUCCESS, result=result)
    payload = exec_result.to_dict()

    assert payload["result"]["summary"] == "Computed stats."
    assert payload["result"]["artifacts"][0]["path"] == "/tmp/stats.png"
    json.dumps(payload)


def _serialized_tool_payload(artifact_path: str) -> dict:
    return {
        "kind": "statistics",
        "summary": "Computed stats.",
        "data": {"mean": 1.5},
        "artifacts": [
            {
                "kind": "image",
                "path": artifact_path,
                "mime_type": "image/png",
                "description": "Stats plot",
            }
        ],
    }


def test_persist_subprocess_artifacts_copies_files_out_of_temp_dir(tmp_path):
    artifact_dir = tmp_path / "subprocess" / "artifacts"
    artifact_dir.mkdir(parents=True)
    source_path = artifact_dir / "stats.png"
    source_path.write_bytes(b"subprocess-png")

    result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        result=_serialized_tool_payload(str(source_path)),
        formatted_output="stale formatted output",
    )

    persisted = _persist_subprocess_artifacts(
        result,
        host_artifact_dir=artifact_dir,
    )
    persisted_path = Path(persisted.result["artifacts"][0]["path"])

    assert persisted_path != source_path
    assert persisted_path.read_bytes() == b"subprocess-png"

    shutil.rmtree(artifact_dir.parent)
    assert persisted_path.exists()
    assert str(persisted_path) in persisted.formatted_output


def test_persist_docker_artifacts_remaps_container_paths(tmp_path):
    artifact_dir = tmp_path / "docker" / "artifacts"
    artifact_dir.mkdir(parents=True)
    host_path = artifact_dir / "stats.png"
    host_path.write_bytes(b"docker-png")

    result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        result=_serialized_tool_payload("/io/artifacts/stats.png"),
        formatted_output="stale formatted output",
    )

    persisted = _persist_docker_artifacts(
        result,
        container_artifact_dir="/io/artifacts",
        host_artifact_dir=artifact_dir,
    )
    persisted_path = Path(persisted.result["artifacts"][0]["path"])

    assert persisted_path != host_path
    assert persisted_path.read_bytes() == b"docker-png"
    assert not str(persisted_path).startswith("/io/artifacts")

    shutil.rmtree(artifact_dir.parent)
    assert persisted_path.exists()
    assert str(persisted_path) in persisted.formatted_output


def test_local_backend_formats_tool_payload_with_artifacts():
    backend = LocalBackend()

    def fake_tool():
        return ToolPayload(
            kind="patterns",
            summary="Detected 3 peaks.",
            data={"count": 3},
            artifacts=[
                ArtifactRef(
                    kind="image",
                    path="/tmp/peaks.png",
                    mime_type="image/png",
                    description="Peak detection plot",
                )
            ],
        )

    result = backend.execute(
        "detect_peaks_with_data",
        lambda: fake_tool(),
        {},
        ExecutionContext(),
    )

    assert result.success is True
    assert "Detected 3 peaks." in result.formatted_output
    assert "Artifacts:" in result.formatted_output
    assert "/tmp/peaks.png" in result.formatted_output


def test_describe_sandbox_backend_docker_reports_missing_image(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    monkeypatch.setattr(executor_mod.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd, capture_output=True, text=True, timeout=5):
        if cmd[:3] == ["docker", "version", "--format"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="29.3.1\n", stderr="")
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="Error: No such image: ts-agents-sandbox:latest\n",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(executor_mod.subprocess, "run", fake_run)

    status = describe_sandbox_backend("docker")

    assert status["backend"] == "docker"
    assert status["available"] is False
    assert status["reason"] == "Docker image 'ts-agents-sandbox:latest' is not available locally."
    assert "Build or pull 'ts-agents-sandbox:latest'" in status["suggested_fix"]
    assert status["details"]["image"] == "ts-agents-sandbox:latest"
    assert "No such image" in status["details"]["image_probe_error"]


def test_describe_sandbox_backend_docker_honors_context_image_override(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    seen_images = []

    monkeypatch.setattr(executor_mod.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd, capture_output=True, text=True, timeout=5):
        if cmd[:3] == ["docker", "version", "--format"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="29.3.1\n", stderr="")
        if cmd[:3] == ["docker", "image", "inspect"]:
            seen_images.append(cmd[3])
            return subprocess.CompletedProcess(cmd, 0, stdout="sha256:custom\n", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(executor_mod.subprocess, "run", fake_run)

    status = describe_sandbox_backend(
        "docker",
        context=ExecutionContext(sandbox_mode="docker", docker_image="custom:ready"),
    )

    assert status["available"] is True
    assert status["details"]["image"] == "custom:ready"
    assert status["details"]["image_id"] == "sha256:custom"
    assert seen_images == ["custom:ready"]


def test_describe_sandbox_backend_docker_honors_backend_image_default(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    seen_images = []

    monkeypatch.delenv("TS_AGENTS_DOCKER_IMAGE", raising=False)
    monkeypatch.setattr(executor_mod.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd, capture_output=True, text=True, timeout=5):
        if cmd[:3] == ["docker", "version", "--format"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="29.3.1\n", stderr="")
        if cmd[:3] == ["docker", "image", "inspect"]:
            seen_images.append(cmd[3])
            return subprocess.CompletedProcess(cmd, 0, stdout="sha256:backend\n", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(executor_mod.subprocess, "run", fake_run)

    status = describe_sandbox_backend(
        "docker",
        backend=executor_mod.DockerBackend(image="backend:ready"),
    )

    assert status["available"] is True
    assert status["details"]["image"] == "backend:ready"
    assert status["details"]["image_id"] == "sha256:backend"
    assert seen_images == ["backend:ready"]


def test_tool_executor_passes_context_and_backend_to_readiness_probe(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    class FakeDockerBackend:
        image = "backend:ready"

        def is_available(self):
            return True

        def execute(self, tool_name, func, params, context):
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                result={"kind": "analysis", "data": {}, "artifacts": []},
                formatted_output="ok",
                metadata={"backend": "docker"},
            )

    executor = ToolExecutor(
        default_backend=SandboxMode.LOCAL,
        backends={
            SandboxMode.LOCAL: LocalBackend(),
            SandboxMode.DOCKER: FakeDockerBackend(),
            SandboxMode.DAYTONA: executor_mod.DaytonaBackend(),
            SandboxMode.MODAL: executor_mod.ModalBackend(),
            SandboxMode.SUBPROCESS: executor_mod.SubprocessBackend(),
        },
    )
    observed = {}

    def fake_describe(mode, context=None, backend=None):
        observed["mode"] = mode
        observed["context"] = context
        observed["backend"] = backend
        return {
            "backend": mode.value,
            "available": True,
            "reason": None,
            "suggested_fix": None,
            "requirements": [],
            "details": {},
            "description": "backend",
        }

    monkeypatch.setattr(executor_mod, "describe_sandbox_backend", fake_describe)

    context = ExecutionContext(sandbox_mode="docker", docker_image="custom:ready")
    result = executor.execute(
        "describe_series",
        {"series": [1, 2, 3]},
        context=context,
    )

    assert result.success is True
    assert observed["mode"] == SandboxMode.DOCKER
    assert observed["context"] is context
    assert observed["backend"] is executor.backends[SandboxMode.DOCKER]


def test_tool_executor_returns_backend_unavailable_without_fallback(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    executor = ToolExecutor(default_backend=SandboxMode.LOCAL)

    def fake_describe(mode, context=None, backend=None):
        if mode == SandboxMode.DOCKER:
            return {
                "backend": "docker",
                "available": False,
                "reason": "Docker CLI not found.",
                "suggested_fix": "Install Docker or run with --sandbox local.",
                "requirements": [],
                "details": {},
                "description": "Containerized execution through Docker.",
            }
        return {
            "backend": mode.value,
            "available": True,
            "reason": None,
            "suggested_fix": None,
            "requirements": [],
            "details": {},
            "description": "backend",
        }

    monkeypatch.setattr(executor_mod, "describe_sandbox_backend", fake_describe)
    monkeypatch.setattr(executor.backends[SandboxMode.DOCKER], "is_available", lambda: False)

    result = executor.execute(
        "describe_series",
        {"series": [1, 2, 3]},
        context=ExecutionContext(sandbox_mode="docker"),
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == ToolErrorCode.BACKEND_UNAVAILABLE
    assert result.metadata["backend_requested"] == "docker"
    assert result.metadata["backend_actual"] is None
    assert result.metadata["fallback_allowed"] is False


def test_tool_executor_can_fallback_to_local_when_docker_image_is_missing(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    executor = ToolExecutor(default_backend=SandboxMode.LOCAL)

    def fake_describe(mode, context=None, backend=None):
        if mode == SandboxMode.DOCKER:
            return {
                "backend": "docker",
                "available": False,
                "reason": "Docker image 'ts-agents-sandbox:latest' is not available locally.",
                "suggested_fix": "Build or pull 'ts-agents-sandbox:latest', set TS_AGENTS_DOCKER_IMAGE to an available image, or run with --sandbox local.",
                "requirements": [],
                "details": {"image": "ts-agents-sandbox:latest"},
                "description": "Containerized execution through Docker.",
            }
        return {
            "backend": mode.value,
            "available": True,
            "reason": None,
            "suggested_fix": None,
            "requirements": [],
            "details": {},
            "description": "backend",
        }

    monkeypatch.setattr(executor_mod, "describe_sandbox_backend", fake_describe)
    monkeypatch.setattr(executor.backends[SandboxMode.DOCKER], "is_available", lambda: True)
    monkeypatch.setattr(executor.backends[SandboxMode.LOCAL], "is_available", lambda: True)

    result = executor.execute(
        "describe_series",
        {"series": [1, 2, 3]},
        context=ExecutionContext(
            sandbox_mode="docker",
            allow_fallback=True,
            fallback_backend="local",
        ),
    )

    assert result.success is True
    assert result.metadata["backend_requested"] == "docker"
    assert result.metadata["backend_actual"] == "local"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_allowed"] is True


def test_tool_executor_can_fallback_to_local_when_allowed(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    executor = ToolExecutor(default_backend=SandboxMode.LOCAL)

    def fake_describe(mode, context=None, backend=None):
        if mode == SandboxMode.DOCKER:
            return {
                "backend": "docker",
                "available": False,
                "reason": "Docker CLI not found.",
                "suggested_fix": "Install Docker or run with --sandbox local.",
                "requirements": [],
                "details": {},
                "description": "Containerized execution through Docker.",
            }
        return {
            "backend": mode.value,
            "available": True,
            "reason": None,
            "suggested_fix": None,
            "requirements": [],
            "details": {},
            "description": "backend",
        }

    monkeypatch.setattr(executor_mod, "describe_sandbox_backend", fake_describe)
    monkeypatch.setattr(executor.backends[SandboxMode.DOCKER], "is_available", lambda: False)

    result = executor.execute(
        "describe_series",
        {"series": [1, 2, 3]},
        context=ExecutionContext(
            sandbox_mode="docker",
            allow_fallback=True,
            fallback_backend="local",
        ),
    )

    assert result.success is True
    assert result.metadata["backend_requested"] == "docker"
    assert result.metadata["backend_actual"] == "local"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_allowed"] is True


def test_tool_executor_rejects_unavailable_fallback_backend(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    executor = ToolExecutor(default_backend=SandboxMode.LOCAL)

    def fake_describe(mode, context=None, backend=None):
        if mode == SandboxMode.DOCKER:
            return {
                "backend": "docker",
                "available": False,
                "reason": "Docker CLI not found.",
                "suggested_fix": "Install Docker or run with --sandbox local.",
                "requirements": [],
                "details": {},
                "description": "Containerized execution through Docker.",
            }
        return {
            "backend": mode.value,
            "available": True,
            "reason": None,
            "suggested_fix": None,
            "requirements": [],
            "details": {},
            "description": "backend",
        }

    monkeypatch.setattr(executor_mod, "describe_sandbox_backend", fake_describe)
    monkeypatch.setattr(executor.backends[SandboxMode.DOCKER], "is_available", lambda: False)
    monkeypatch.setattr(executor.backends[SandboxMode.LOCAL], "is_available", lambda: False)

    result = executor.execute(
        "describe_series",
        {"series": [1, 2, 3]},
        context=ExecutionContext(
            sandbox_mode="docker",
            allow_fallback=True,
            fallback_backend="local",
        ),
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == ToolErrorCode.BACKEND_UNAVAILABLE
    assert result.metadata["backend_actual"] is None
    assert result.metadata["fallback_used"] is False


def test_staged_artifact_bundle_keeps_manifest_and_report_when_models_exceed_total_cap(tmp_path):
    from ts_agents.tools.artifact_staging import collect_staged_artifact_files

    output_dir = tmp_path / "panel"
    (output_dir / "models" / "lightgbm").mkdir(parents=True)
    (output_dir / "models" / "lightgbm" / "model.pkl").write_bytes(b"x" * 900)
    (output_dir / "models" / "nhits").mkdir(parents=True)
    (output_dir / "models" / "nhits" / "weights.ckpt").write_bytes(b"y" * 900)
    (output_dir / "metrics.json").write_text('{"smape": 1.0}', encoding="utf-8")
    (output_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    (output_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    payload = {}
    staged = collect_staged_artifact_files(
        output_dir,
        payload,
        max_file_bytes=None,
        max_total_bytes=1000,
        file_limit_env="TEST_FILE_LIMIT",
        total_limit_env="TEST_TOTAL_LIMIT",
    )

    staged_paths = [item["relative_path"] for item in staged]
    # models/ sorts first alphabetically, but it is bundled last so the
    # manifest, report and metrics always make it back to the host.
    assert staged_paths[:3] == ["metrics.json", "report.md", "run_manifest.json"]
    assert staged_paths[3:] == ["models/lightgbm/model.pkl"]
    assert len(payload["warnings"]) == 1
    assert "models/nhits/weights.ckpt" in payload["warnings"][0]
    assert "Set TEST_TOTAL_LIMIT to override." in payload["warnings"][0]
def test_docker_relocation_preserves_model_hierarchy_and_metadata(tmp_path):
    from ts_agents.workflows.executor import _rewrite_docker_workflow_output_paths
    root = tmp_path / "sandbox"
    refs = []
    for name in ("models/lightgbm/model.pkl", "models/nhits/model.pkl", "run_manifest.json"):
        path = root / "run" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        refs.append({"kind": "file", "path": "/io/artifacts/run/" + name})
    result = ExecutionResult(status=ExecutionStatus.SUCCESS, result={
        "artifacts": refs, "data": {"output_dir": "/io/artifacts/run",
        "manifest_path": "/io/artifacts/run/run_manifest.json"}})
    _persist_docker_artifacts(result, container_artifact_dir="/io/artifacts", host_artifact_dir=root)
    shutil.rmtree(root)
    dest = tmp_path / "output"
    _rewrite_docker_workflow_output_paths(result, str(dest))
    assert (dest / "models/lightgbm/model.pkl").read_text() == "models/lightgbm/model.pkl"
    assert (dest / "models/nhits/model.pkl").read_text() == "models/nhits/model.pkl"
    assert result.result["data"]["manifest_path"] == str(dest / "run_manifest.json")
    assert len({ref["path"] for ref in result.result["artifacts"]}) == 3


def test_tool_limits_and_daytona_extras_reach_backend_without_mutating_context(monkeypatch):
    from ts_agents.tools import executor as module
    calls = []
    class Backend:
        def is_available(self): return True
        def execute(self, **kwargs):
            calls.append(kwargs)
            return ExecutionResult(status=ExecutionStatus.SUCCESS, result={})
    monkeypatch.setattr(module, "describe_sandbox_backend", lambda *args, **kwargs: {"available": True})
    monkeypatch.delenv("TS_AGENTS_DAYTONA_INSTALL_EXTRAS", raising=False)
    executor = ToolExecutor()
    executor.backends[SandboxMode.DAYTONA] = Backend()
    context = ExecutionContext(sandbox_mode="daytona", user_approved=True)
    assert executor.execute("forecast_foundation", {"series": [1., 2.]}, context=context).success
    actual = calls[-1]["context"]
    assert (actual.timeout_seconds, actual.memory_mb, actual.disk_mb) == (900, 4096, 2048)
    assert actual.environment["TS_AGENTS_DAYTONA_INSTALL_EXTRAS"] == "foundation"
    assert context.timeout_seconds is None and context.memory_mb is None and context.environment == {}
    explicit = ExecutionContext(sandbox_mode="daytona", user_approved=True, timeout_seconds=120,
        memory_mb=1000, environment={"TS_AGENTS_DAYTONA_INSTALL_EXTRAS": "foundation,viz"})
    assert executor.execute("forecast_foundation", {"series": [1., 2.]}, context=explicit).success
    assert calls[-1]["context"].timeout_seconds == 120
    assert calls[-1]["context"].memory_mb == 1000
    assert calls[-1]["context"].environment == explicit.environment


def test_panel_tool_subprocess_returns_durable_directory_and_manifest(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setenv("TS_AGENTS_TOOL_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    path = tmp_path / "panel.csv"
    pd.DataFrame({"unique_id": ["a"] * 30, "ds": pd.date_range("2024-01-01", periods=30),
                  "y": np.arange(30.) + 1}).to_csv(path, index=False)
    result = ToolExecutor().execute("forecast_panel_from_csv", {"input_path": str(path),
        "freq": "D", "horizon": 2, "season_length": 1, "n_windows": 1},
        context=ExecutionContext(sandbox_mode="subprocess", user_approved=True))
    assert result.success, result.error
    data = result.result["data"]
    root = Path(data["output_dir"])
    assert root.is_dir() and Path(data["manifest_path"]).is_file()
    assert all(Path(ref["path"]).is_file() for ref in result.result["artifacts"])
    manifest = json.loads(Path(data["manifest_path"]).read_text())
    assert manifest["output_dir"] == str(root)


def test_panel_tool_transports_rows_and_restores_daytona_bundle(tmp_path, monkeypatch):
    import pandas as pd
    from ts_agents.tools import executor as module
    from ts_agents.workflows import executor as workflow_module
    from ts_agents.tools.results import serialize_result
    monkeypatch.setenv("TS_AGENTS_TOOL_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("TS_AGENTS_DAYTONA_INSTALL_EXTRAS", raising=False)
    for target in (module, workflow_module):
        monkeypatch.setattr(target, "describe_sandbox_backend", lambda *args, **kwargs: {"available": True})
    path = tmp_path / "panel.csv"
    pd.DataFrame({"unique_id": ["a"] * 30, "ds": pd.date_range("2024-01-01", periods=30),
                  "y": np.arange(30.) + 1}).to_csv(path, index=False)
    observed = {}
    class Backend:
        def is_available(self): return True
        def execute(self, **kwargs):
            observed.update(kwargs)
            params = dict(kwargs["params"])
            assert params["workflow_input"]["kind"] == "panel_input"
            assert len(params["workflow_input"]["records"]) == 30
            path.unlink()  # The sandbox cannot read the host file.
            params["sandbox_artifact_dir"] = str(tmp_path / "remote")
            payload = serialize_result(kwargs["func"](**params))
            shutil.rmtree(tmp_path / "remote")  # Equivalent to deleting the ephemeral sandbox.
            return ExecutionResult(status=ExecutionStatus.SUCCESS, result=payload)
    executor = ToolExecutor()
    executor.backends[SandboxMode.DAYTONA] = Backend()
    result = executor.execute("forecast_panel_from_csv", {"input_path": str(path),
        "freq": "D", "horizon": 2, "season_length": 1, "n_windows": 1},
        context=ExecutionContext(sandbox_mode="daytona", user_approved=True))
    assert result.success, result.error
    assert observed["context"].environment["TS_AGENTS_DAYTONA_INSTALL_EXTRAS"] == "viz"
    assert (observed["context"].timeout_seconds, observed["context"].memory_mb) == (1800, 8192)
    root = Path(result.result["data"]["output_dir"])
    assert (root / "models/seasonal_naive/history.csv").is_file()
    assert all(Path(ref["path"]).is_file() for ref in result.result["artifacts"])
    manifest = json.loads(Path(result.result["data"]["manifest_path"]).read_text())
    assert manifest["output_dir"] == str(root)
    assert all(Path(ref["path"]).is_file() for ref in manifest["artifacts"])


def test_docker_artifact_resolver_rejects_escape_and_symlink(tmp_path):
    from ts_agents.tools.executor import _resolve_docker_artifact_path
    root = tmp_path / "artifacts"
    root.mkdir()
    external = tmp_path / "private"
    external.write_text("outside the sandbox volume")
    (root / "link").symlink_to(external)
    for name in ("/io/artifacts/../private", "/io/artifacts/link"):
        assert _resolve_docker_artifact_path(name, container_artifact_dir="/io/artifacts",
                                             host_artifact_dir=root) is None
