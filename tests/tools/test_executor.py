"""Tests for tool executor serialization and context handling."""

import json
import shutil
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


def test_tool_executor_returns_backend_unavailable_without_fallback(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    executor = ToolExecutor(default_backend=SandboxMode.LOCAL)

    def fake_describe(mode):
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


def test_tool_executor_can_fallback_to_local_when_allowed(monkeypatch):
    import ts_agents.tools.executor as executor_mod

    executor = ToolExecutor(default_backend=SandboxMode.LOCAL)

    def fake_describe(mode):
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

    def fake_describe(mode):
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
