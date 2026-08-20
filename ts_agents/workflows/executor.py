"""Workflow execution abstraction with sandbox parity."""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ts_agents.cli.input_parsing import LabeledStreamInput, SeriesInput
from ts_agents.tools import artifact_staging as _staging
from ts_agents.tools.executor import (
    DockerBackend,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    LocalBackend,
    ModalBackend,
    SandboxMode,
    SubprocessBackend,
    ToolError,
    ToolErrorCode,
    DaytonaBackend,
    describe_sandbox_backend,
)
from ts_agents.tools.results import format_result, serialize_result

from . import get_workflow

_WORKFLOW_PREFIX = "workflow:"
_SANDBOX_ARTIFACT_DIR_ENV = _staging.SANDBOX_ARTIFACT_DIR_ENV
_STAGED_WORKFLOW_ARTIFACTS_KEY = "_ts_agents_staged_workflow_artifacts"
_WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV = _staging.WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV
_WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV = _staging.WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV

_enforce_host_availability_for_backend = _staging.enforce_host_availability_for_backend
_append_payload_warning = _staging.append_payload_warning


def _workflow_target_name(workflow_name: str) -> str:
    return f"{_WORKFLOW_PREFIX}{workflow_name}"


def is_workflow_target(tool_name: str) -> bool:
    """Return whether a sandbox request targets a workflow."""
    return tool_name.startswith(_WORKFLOW_PREFIX)


def _serialize_workflow_input(workflow_input: Any) -> Dict[str, Any]:
    if isinstance(workflow_input, SeriesInput):
        payload = asdict(workflow_input)
        payload["kind"] = "series_input"
        payload["series"] = workflow_input.series.tolist()
        return payload

    if isinstance(workflow_input, LabeledStreamInput):
        payload = asdict(workflow_input)
        payload["kind"] = "labeled_stream_input"
        payload["values"] = workflow_input.values.tolist()
        payload["labels"] = workflow_input.labels.tolist()
        return payload

    raise TypeError(f"Unsupported workflow input type: {type(workflow_input).__name__}")


def _deserialize_workflow_input(payload: Dict[str, Any]) -> Any:
    kind = payload.get("kind")
    if kind == "series_input":
        data = dict(payload)
        data.pop("kind", None)
        data["series"] = np.asarray(data.get("series") or [], dtype=np.float64)
        return SeriesInput(**data)

    if kind == "labeled_stream_input":
        data = dict(payload)
        data.pop("kind", None)
        data["values"] = np.asarray(data.get("values") or [], dtype=np.float64)
        data["labels"] = np.asarray(data.get("labels") or [])
        return LabeledStreamInput(**data)

    raise ValueError(f"Unsupported workflow input payload kind: {kind}")


def _workflow_artifact_bundle_limits() -> Tuple[Optional[int], Optional[int]]:
    return _staging.artifact_bundle_limits(
        (_WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV,),
        (_WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV,),
    )


def _run_serialized_workflow(
    *,
    workflow_name: str,
    workflow_input: Dict[str, Any],
    runner_kwargs: Dict[str, Any],
    use_sandbox_artifact_dir: bool = False,
    sandbox_artifact_dir: Optional[str] = None,
    bundle_sandbox_artifacts: bool = False,
) -> Any:
    workflow = get_workflow(workflow_name)
    resolved_kwargs = dict(runner_kwargs or {})
    staged_output_dir: Optional[Path] = None
    if use_sandbox_artifact_dir:
        artifact_root = sandbox_artifact_dir or os.environ.get(_SANDBOX_ARTIFACT_DIR_ENV)
        if not artifact_root:
            raise RuntimeError(
                "sandbox_artifact_dir or "
                f"{_SANDBOX_ARTIFACT_DIR_ENV} is required when use_sandbox_artifact_dir=true."
            )
        staged_output_dir = Path(artifact_root) / workflow_name
        resolved_kwargs["output_dir"] = str(staged_output_dir)

    resolved_input = _deserialize_workflow_input(workflow_input)
    result = workflow.runner(resolved_input, **resolved_kwargs)
    if staged_output_dir is not None and bundle_sandbox_artifacts:
        return _attach_staged_workflow_artifacts(result, staged_output_dir)
    return result


def _attach_staged_workflow_artifacts(
    result: Any,
    output_dir: Path,
) -> Any:
    payload = serialize_result(result)
    if not isinstance(payload, dict):
        return payload

    max_file_bytes, max_total_bytes = _workflow_artifact_bundle_limits()
    staged_files = _staging.collect_staged_artifact_files(
        output_dir,
        payload,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        file_limit_env=_WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV,
        total_limit_env=_WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV,
    )
    if staged_files:
        payload[_STAGED_WORKFLOW_ARTIFACTS_KEY] = staged_files
    return payload


def _use_staged_workflow_artifact_dir(backend: SandboxMode) -> bool:
    return _staging.use_staged_artifact_dir(backend)


def _bundle_staged_workflow_artifacts(backend: SandboxMode) -> bool:
    return _staging.bundle_staged_artifacts(backend)


def _sandbox_workflow_artifact_dir(
    backend: SandboxMode,
) -> Optional[str]:
    # The Docker path is now unique per run (previously the static
    # /io/artifacts, which collided across concurrent workflow runs).
    return _staging.sandbox_artifact_dir(
        backend, modal_prefix="ts_agents_workflow_artifacts"
    )


def execute_serialized_workflow_request(
    *,
    workflow_name: str,
    kwargs: Dict[str, Any],
    context: Optional[ExecutionContext] = None,
) -> ExecutionResult:
    """Execute a serialized workflow request inside a sandbox runner."""
    context = context or ExecutionContext(sandbox_mode=SandboxMode.LOCAL)
    backend = LocalBackend()
    return backend.execute(
        tool_name=_workflow_target_name(workflow_name),
        func=_run_serialized_workflow,
        params=kwargs,
        context=context,
    )


class WorkflowExecutor:
    """Execute workflows with the same sandbox controls as tools."""

    def __init__(self) -> None:
        self.backends = {
            SandboxMode.LOCAL: LocalBackend(),
            SandboxMode.DOCKER: DockerBackend(),
            SandboxMode.DAYTONA: DaytonaBackend(),
            SandboxMode.MODAL: ModalBackend(),
            SandboxMode.SUBPROCESS: SubprocessBackend(),
        }

    def execute(
        self,
        workflow_name: str,
        workflow_input: Any,
        runner_kwargs: Dict[str, Any],
        *,
        context: Optional[ExecutionContext] = None,
    ) -> ExecutionResult:
        context = context or ExecutionContext(sandbox_mode=SandboxMode.LOCAL)

        try:
            workflow = get_workflow(workflow_name)
        except KeyError:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=ToolError(
                    code=ToolErrorCode.NOT_FOUND,
                    message=f"Unknown workflow '{workflow_name}'.",
                    recoverable=False,
                    tool_name=workflow_name,
                ),
                metadata={
                    "workflow_name": workflow_name,
                    "backend_requested": getattr(context.sandbox_mode, "value", str(context.sandbox_mode)),
                    "backend_actual": None,
                },
            )

        requested_backend = context.sandbox_mode
        actual_backend = context.sandbox_mode
        backend = self.backends.get(requested_backend)
        requested_status = describe_sandbox_backend(
            requested_backend,
            context=context,
            backend=backend,
        )
        fallback_backend = context.fallback_backend or SandboxMode.LOCAL

        if backend is None or not requested_status["available"] or not backend.is_available():
            if not context.allow_fallback:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=ToolError(
                        code=ToolErrorCode.BACKEND_UNAVAILABLE,
                        message=(
                            f"Requested backend '{requested_backend.value}' is unavailable and fallback is not allowed."
                        ),
                        recoverable=True,
                        hint=requested_status.get("suggested_fix")
                        or f"Run `ts-agents sandbox doctor {requested_backend.value}` or retry with --allow-fallback.",
                        tool_name=workflow_name,
                        details={
                            "backend_requested": requested_backend.value,
                            "backend_status": requested_status,
                            "fallback_allowed": False,
                        },
                    ),
                    metadata={
                        "workflow_name": workflow_name,
                        "backend_requested": requested_backend.value,
                        "backend_actual": None,
                        "fallback_allowed": False,
                        "fallback_used": False,
                        "backend_status": requested_status,
                    },
                )

            backend = self.backends.get(fallback_backend)
            fallback_status = describe_sandbox_backend(
                fallback_backend,
                context=context,
                backend=backend,
            )
            if backend is None or not fallback_status["available"] or not backend.is_available():
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=ToolError(
                        code=ToolErrorCode.BACKEND_UNAVAILABLE,
                        message=(
                            f"Requested backend '{requested_backend.value}' is unavailable and fallback backend "
                            f"'{fallback_backend.value}' is also unavailable."
                        ),
                        recoverable=True,
                        hint=fallback_status.get("suggested_fix") or requested_status.get("suggested_fix"),
                        tool_name=workflow_name,
                        details={
                            "backend_requested": requested_backend.value,
                            "requested_backend_status": requested_status,
                            "fallback_backend": fallback_backend.value,
                            "fallback_backend_status": fallback_status,
                            "fallback_allowed": True,
                        },
                    ),
                    metadata={
                        "workflow_name": workflow_name,
                        "backend_requested": requested_backend.value,
                        "backend_actual": None,
                        "fallback_allowed": True,
                        "fallback_backend": fallback_backend.value,
                        "fallback_used": False,
                        "backend_status": requested_status,
                    },
                )

            actual_backend = fallback_backend

        availability = workflow.availability()
        if (
            _enforce_host_availability_for_backend(actual_backend)
            and not availability.get("available", True)
        ):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=ToolError(
                    code=ToolErrorCode.DEPENDENCY_ERROR,
                    message=availability.get("install_hint")
                    or f"Workflow '{workflow_name}' is unavailable in the current environment.",
                    recoverable=False,
                    tool_name=workflow_name,
                    details={"availability": availability},
                ),
                metadata={
                    "workflow_name": workflow_name,
                    "backend_requested": requested_backend.value,
                    "backend_actual": actual_backend.value,
                    "fallback_allowed": context.allow_fallback,
                    "fallback_backend": fallback_backend.value if context.allow_fallback else None,
                    "fallback_used": actual_backend != requested_backend,
                    "availability": availability,
                },
            )

        request_payload = {
            "workflow_name": workflow_name,
            "workflow_input": _serialize_workflow_input(workflow_input),
            "runner_kwargs": dict(runner_kwargs or {}),
            "use_sandbox_artifact_dir": _use_staged_workflow_artifact_dir(actual_backend),
            "sandbox_artifact_dir": _sandbox_workflow_artifact_dir(actual_backend),
            "bundle_sandbox_artifacts": _bundle_staged_workflow_artifacts(actual_backend),
        }
        requested_output_dir = request_payload["runner_kwargs"].get("output_dir")

        result = backend.execute(
            tool_name=_workflow_target_name(workflow_name),
            func=_run_serialized_workflow,
            params=request_payload,
            context=context,
        )
        result.metadata = {
            **(result.metadata or {}),
            "workflow_name": workflow_name,
            "backend_requested": requested_backend.value,
            "backend_actual": actual_backend.value,
            "fallback_used": actual_backend != requested_backend,
            "fallback_allowed": context.allow_fallback,
            "fallback_backend": fallback_backend.value if context.allow_fallback else None,
            "availability": availability,
        }

        if result.success and actual_backend == SandboxMode.DOCKER and requested_output_dir:
            _rewrite_docker_workflow_output_paths(result, requested_output_dir)
        elif result.success and actual_backend in {SandboxMode.DAYTONA, SandboxMode.MODAL}:
            _materialize_remote_workflow_output_paths(result, requested_output_dir)

        return result


def _rewrite_docker_workflow_output_paths(
    result: ExecutionResult,
    requested_output_dir: str,
) -> None:
    payload = result.result
    if not isinstance(payload, dict):
        return

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return

    destination_dir = Path(requested_output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    rewritten_paths: Dict[str, Path] = {}

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        source = Path(str(artifact.get("path", "")))
        if not source.exists():
            continue
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        rewritten_paths[str(source)] = destination
        artifact["path"] = str(destination)

    data = payload.get("data")
    if isinstance(data, dict):
        data["output_dir"] = str(destination_dir)
        _rewrite_workflow_run_metadata_paths(data, destination_dir, rewritten_paths)

    result.formatted_output = format_result(payload)


def _rewrite_workflow_run_metadata_paths(
    data: Dict[str, Any],
    destination_dir: Path,
    rewritten_paths: Dict[str, Path],
) -> None:
    manifest_path = data.get("manifest_path")
    if isinstance(manifest_path, str):
        destination_manifest = rewritten_paths.get(manifest_path)
        if destination_manifest is not None:
            data["manifest_path"] = str(destination_manifest)

    run_metadata = data.get("run")
    if isinstance(run_metadata, dict):
        run_metadata["output_dir"] = str(destination_dir)
        run_manifest_path = run_metadata.get("manifest_path")
        if isinstance(run_manifest_path, str):
            destination_manifest = rewritten_paths.get(run_manifest_path)
            if destination_manifest is not None:
                run_metadata["manifest_path"] = str(destination_manifest)


def _materialize_remote_workflow_output_paths(
    result: ExecutionResult,
    requested_output_dir: Optional[str],
) -> None:
    payload = result.result
    if not isinstance(payload, dict):
        return

    staged_files = payload.pop(_STAGED_WORKFLOW_ARTIFACTS_KEY, None)
    if not isinstance(staged_files, list) or not staged_files:
        return

    # Resolved once for the whole bundle. When no output dir was requested a
    # single temp dir is created lazily; previously mkdtemp ran per staged
    # file, scattering one bundle across as many temp dirs as files.
    destination_root: Optional[Path] = (
        Path(requested_output_dir).resolve() if requested_output_dir else None
    )
    destination_dir: Optional[Path] = None
    rewritten_paths: Dict[str, Path] = {}
    for staged_file in staged_files:
        if not isinstance(staged_file, dict):
            continue
        relative_path = staged_file.get("relative_path")
        content_base64 = staged_file.get("content_base64")
        source_path = staged_file.get("source_path")
        if not isinstance(relative_path, str) or not isinstance(content_base64, str):
            continue
        candidate = Path(relative_path)
        if candidate.is_absolute():
            _append_payload_warning(
                payload,
                f"Skipped restoring remote artifact '{relative_path}' because absolute paths are not allowed.",
            )
            continue
        if destination_root is None:
            destination_root = Path(
                tempfile.mkdtemp(prefix="ts_agents_workflow_output_")
            ).resolve()
        destination = destination_root / candidate
        resolved_destination = destination.resolve(strict=False)
        try:
            resolved_destination.relative_to(destination_root)
        except ValueError:
            _append_payload_warning(
                payload,
                f"Skipped restoring remote artifact '{relative_path}' because it escapes the output directory.",
            )
            continue
        try:
            content = base64.b64decode(content_base64.encode("ascii"))
        except (ValueError, binascii.Error):
            _append_payload_warning(
                payload,
                f"Skipped restoring remote artifact '{relative_path}' because its payload was not valid base64.",
            )
            continue
        if destination_dir is None:
            destination_dir = destination_root
            destination_dir.mkdir(parents=True, exist_ok=True)
        try:
            _staging.write_bytes_atomically(resolved_destination, content)
        except OSError as exc:
            _append_payload_warning(
                payload,
                f"Skipped restoring remote artifact '{relative_path}' because it could not be written: {exc}",
            )
            continue
        if isinstance(source_path, str):
            rewritten_paths[source_path] = resolved_destination

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            source_path = artifact.get("path")
            if not isinstance(source_path, str):
                continue
            destination = rewritten_paths.get(source_path)
            if destination is not None:
                artifact["path"] = str(destination)
            else:
                _append_payload_warning(
                    payload,
                    f"Artifact '{source_path}' was not staged and remains inaccessible on the host.",
                )

    if not rewritten_paths:
        result.formatted_output = format_result(payload)
        return

    data = payload.get("data")
    if isinstance(data, dict) and destination_dir is not None:
        data["output_dir"] = str(destination_dir)
        _rewrite_workflow_run_metadata_paths(data, destination_dir, rewritten_paths)

    result.formatted_output = format_result(payload)


_DEFAULT_EXECUTOR: Optional[WorkflowExecutor] = None


def get_executor() -> WorkflowExecutor:
    global _DEFAULT_EXECUTOR
    if _DEFAULT_EXECUTOR is None:
        _DEFAULT_EXECUTOR = WorkflowExecutor()
    return _DEFAULT_EXECUTOR


def execute_workflow(
    workflow_name: str,
    workflow_input: Any,
    runner_kwargs: Dict[str, Any],
    *,
    context: Optional[ExecutionContext] = None,
) -> ExecutionResult:
    """Execute a workflow using the default workflow executor."""
    return get_executor().execute(
        workflow_name,
        workflow_input,
        runner_kwargs,
        context=context,
    )
