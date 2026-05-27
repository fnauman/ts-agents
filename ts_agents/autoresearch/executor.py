"""Sandbox-aware execution for autoresearch loops."""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple
import uuid

from ts_agents.cli.output import to_jsonable
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

from .registry import get_loop
from .runner import run_autoresearch_loop

_AUTORESEARCH_PREFIX = "autoresearch:"
_SANDBOX_ARTIFACT_DIR_ENV = "TS_AGENTS_TOOL_ARTIFACT_DIR"
_STAGED_AUTORESEARCH_ARTIFACTS_KEY = "_ts_agents_staged_autoresearch_artifacts"
_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES_ENV = "TS_AGENTS_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES"
_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES_ENV = "TS_AGENTS_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES"
_WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV = "TS_AGENTS_WORKFLOW_ARTIFACT_MAX_FILE_BYTES"
_WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV = "TS_AGENTS_WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES"
_DEFAULT_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES = 16 * 1024 * 1024
_DEFAULT_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES = 64 * 1024 * 1024


def _autoresearch_target_name(loop_name: str) -> str:
    return f"{_AUTORESEARCH_PREFIX}{loop_name}"


def is_autoresearch_target(tool_name: str) -> bool:
    """Return whether a sandbox request targets an autoresearch loop."""
    return tool_name.startswith(_AUTORESEARCH_PREFIX)


def _parse_artifact_limit(var_names: tuple[str, ...], default: int) -> Optional[int]:
    for var_name in var_names:
        raw = os.environ.get(var_name)
        if raw is None or not raw.strip():
            continue
        try:
            value = int(raw.strip())
        except ValueError:
            return default
        if value <= 0:
            return None
        return value
    return default


def _autoresearch_artifact_bundle_limits() -> Tuple[Optional[int], Optional[int]]:
    return (
        _parse_artifact_limit(
            (_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES_ENV, _WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV),
            _DEFAULT_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES,
        ),
        _parse_artifact_limit(
            (_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES_ENV, _WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV),
            _DEFAULT_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES,
        ),
    )


def _run_serialized_autoresearch(
    *,
    loop_name: str,
    options: Dict[str, Any],
    use_sandbox_artifact_dir: bool = False,
    sandbox_artifact_dir: Optional[str] = None,
    bundle_sandbox_artifacts: bool = False,
) -> Any:
    resolved_options = dict(options or {})
    staged_output_dir: Optional[Path] = None
    if use_sandbox_artifact_dir:
        artifact_root = (
            sandbox_artifact_dir
            or resolved_options.get("_sandbox_artifact_dir")
            or os.environ.get(_SANDBOX_ARTIFACT_DIR_ENV)
        )
        if not artifact_root:
            raise RuntimeError(
                "sandbox_artifact_dir or "
                f"{_SANDBOX_ARTIFACT_DIR_ENV} is required when use_sandbox_artifact_dir=true."
            )
        staged_output_dir = Path(artifact_root) / loop_name
        resolved_options["output_dir"] = str(staged_output_dir)

    result = run_autoresearch_loop(loop_name=loop_name, **resolved_options)
    if staged_output_dir is not None and bundle_sandbox_artifacts:
        return _attach_staged_autoresearch_artifacts(result, staged_output_dir)
    return result


def _attach_staged_autoresearch_artifacts(result: Any, output_dir: Path) -> Any:
    payload = serialize_result(result)
    if not isinstance(payload, dict):
        return payload

    max_file_bytes, max_total_bytes = _autoresearch_artifact_bundle_limits()
    total_bytes = 0
    staged_files = []
    if output_dir.exists():
        for file_path in sorted(output_dir.rglob("*")):
            if not file_path.is_file():
                continue
            file_size = file_path.stat().st_size
            relative_path = file_path.relative_to(output_dir).as_posix()
            if max_file_bytes is not None and file_size > max_file_bytes:
                _append_payload_warning(
                    payload,
                    f"Skipped remote artifact staging for '{relative_path}' because it exceeds {max_file_bytes} bytes. "
                    f"Set {_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES_ENV} to override.",
                )
                continue
            if max_total_bytes is not None and total_bytes + file_size > max_total_bytes:
                _append_payload_warning(
                    payload,
                    f"Skipped remote artifact staging for '{relative_path}' because total bundle would exceed {max_total_bytes} bytes. "
                    f"Set {_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES_ENV} to override.",
                )
                continue
            content = file_path.read_bytes()
            total_bytes += len(content)
            staged_files.append(
                {
                    "source_path": str(file_path.resolve()),
                    "relative_path": relative_path,
                    "content_base64": base64.b64encode(content).decode("ascii"),
                }
            )

    if staged_files:
        payload[_STAGED_AUTORESEARCH_ARTIFACTS_KEY] = staged_files
    return payload


def execute_serialized_autoresearch_request(
    *,
    loop_name: str,
    kwargs: Dict[str, Any],
    context: Optional[ExecutionContext] = None,
) -> ExecutionResult:
    """Execute a serialized autoresearch request inside a sandbox runner."""
    context = context or ExecutionContext(sandbox_mode=SandboxMode.LOCAL)
    return LocalBackend().execute(
        tool_name=_autoresearch_target_name(loop_name),
        func=_run_serialized_autoresearch,
        params=kwargs,
        context=context,
    )


class AutoresearchExecutor:
    """Execute autoresearch loops with the standard sandbox backends."""

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
        loop_name: str,
        options: Dict[str, Any],
        *,
        context: Optional[ExecutionContext] = None,
    ) -> ExecutionResult:
        context = context or ExecutionContext(sandbox_mode=SandboxMode.LOCAL)
        try:
            loop_definition = get_loop(loop_name)
        except KeyError as exc:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=ToolError(
                    code=ToolErrorCode.NOT_FOUND,
                    message=str(exc),
                    recoverable=False,
                    tool_name=loop_name,
                ),
                metadata={"loop_name": loop_name},
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
                        hint=requested_status.get("suggested_fix"),
                        tool_name=loop_name,
                        details={
                            "backend_requested": requested_backend.value,
                            "backend_status": requested_status,
                            "fallback_allowed": False,
                        },
                    ),
                    metadata={
                        "loop_name": loop_name,
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
                        tool_name=loop_name,
                    ),
                    metadata={
                        "loop_name": loop_name,
                        "backend_requested": requested_backend.value,
                        "backend_actual": None,
                        "fallback_allowed": True,
                        "fallback_backend": fallback_backend.value,
                        "fallback_used": False,
                        "backend_status": requested_status,
                    },
                )
            actual_backend = fallback_backend

        if actual_backend == SandboxMode.DAYTONA and loop_definition.required_extras:
            env = dict(context.environment or {})
            env.setdefault(
                "TS_AGENTS_DAYTONA_INSTALL_EXTRAS",
                ",".join(loop_definition.required_extras),
            )
            context.environment = env

        requested_output_dir = dict(options or {}).get("output_dir")
        request_payload = {
            "loop_name": loop_name,
            "options": dict(options or {}),
            "use_sandbox_artifact_dir": _use_staged_autoresearch_artifact_dir(actual_backend),
            "sandbox_artifact_dir": _sandbox_autoresearch_artifact_dir(actual_backend),
            "bundle_sandbox_artifacts": _bundle_staged_autoresearch_artifacts(actual_backend),
        }

        result = backend.execute(
            tool_name=_autoresearch_target_name(loop_name),
            func=_run_serialized_autoresearch,
            params=request_payload,
            context=context,
        )
        result.metadata = {
            **(result.metadata or {}),
            "loop_name": loop_name,
            "backend_requested": requested_backend.value,
            "backend_actual": actual_backend.value,
            "fallback_used": actual_backend != requested_backend,
            "fallback_allowed": context.allow_fallback,
            "fallback_backend": fallback_backend.value if context.allow_fallback else None,
        }

        if result.success and actual_backend == SandboxMode.DOCKER and requested_output_dir:
            _materialize_existing_artifacts(result, requested_output_dir)
        elif result.success and actual_backend in {SandboxMode.DAYTONA, SandboxMode.MODAL}:
            _materialize_remote_autoresearch_output(result, requested_output_dir)
        return result


def _use_staged_autoresearch_artifact_dir(backend: SandboxMode) -> bool:
    return backend in {SandboxMode.DOCKER, SandboxMode.DAYTONA, SandboxMode.MODAL}


def _bundle_staged_autoresearch_artifacts(backend: SandboxMode) -> bool:
    return backend in {SandboxMode.DAYTONA, SandboxMode.MODAL}


def _sandbox_autoresearch_artifact_dir(backend: SandboxMode) -> Optional[str]:
    if backend == SandboxMode.DOCKER:
        return "/io/artifacts"
    if backend == SandboxMode.DAYTONA:
        return f".ts_agents_io/artifacts/{uuid.uuid4().hex[:8]}"
    if backend == SandboxMode.MODAL:
        return f"/tmp/ts_agents_autoresearch_artifacts/{uuid.uuid4().hex[:8]}"
    return None


def _append_payload_warning(payload: Dict[str, Any], warning: str) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        payload["warnings"] = warnings
    if warning not in warnings:
        warnings.append(warning)


def _materialize_existing_artifacts(result: ExecutionResult, requested_output_dir: str) -> None:
    payload = result.result
    if not isinstance(payload, dict):
        return
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return

    destination_dir = Path(requested_output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    rewritten: dict[str, Path] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        source_path = artifact.get("path")
        if not isinstance(source_path, str):
            continue
        source = Path(source_path)
        if not source.exists():
            continue
        destination = destination_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        rewritten[source_path] = destination
        artifact["path"] = str(destination)

    _rewrite_payload_paths(payload, destination_dir, rewritten)
    result.formatted_output = format_result(payload)


def _materialize_remote_autoresearch_output(
    result: ExecutionResult,
    requested_output_dir: Optional[str],
) -> None:
    payload = result.result
    if not isinstance(payload, dict):
        return
    staged_files = payload.pop(_STAGED_AUTORESEARCH_ARTIFACTS_KEY, None)
    if not isinstance(staged_files, list) or not staged_files:
        return

    destination_root = (
        Path(requested_output_dir).resolve()
        if requested_output_dir
        else Path(tempfile.mkdtemp(prefix="ts_agents_autoresearch_output_")).resolve()
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    rewritten: dict[str, Path] = {}
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
        destination = (destination_root / candidate).resolve(strict=False)
        try:
            destination.relative_to(destination_root)
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
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        if isinstance(source_path, str):
            rewritten[source_path] = destination

    _rewrite_remote_artifact_refs(payload, rewritten)
    _rewrite_payload_paths(payload, destination_root, rewritten)
    result.formatted_output = format_result(payload)


def _rewrite_remote_artifact_refs(payload: Dict[str, Any], rewritten: dict[str, Path]) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        source_path = artifact.get("path")
        if not isinstance(source_path, str):
            continue
        destination = rewritten.get(source_path)
        if destination is not None:
            artifact["path"] = str(destination)
        else:
            _append_payload_warning(
                payload,
                f"Artifact '{source_path}' was not staged and remains inaccessible on the host.",
            )


def _rewrite_payload_paths(
    payload: Dict[str, Any],
    destination_dir: Path,
    rewritten: dict[str, Path],
) -> None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return
    data["output_dir"] = str(destination_dir)
    manifest_path = data.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path in rewritten:
        data["manifest_path"] = str(rewritten[manifest_path])
    elif isinstance(manifest_path, str):
        data["manifest_path"] = str(destination_dir / Path(manifest_path).name)

    best_config = data.get("best_config")
    if best_config is not None:
        data["best_config"] = to_jsonable(best_config)


_DEFAULT_EXECUTOR: Optional[AutoresearchExecutor] = None


def get_executor() -> AutoresearchExecutor:
    """Return the default autoresearch executor."""
    global _DEFAULT_EXECUTOR
    if _DEFAULT_EXECUTOR is None:
        _DEFAULT_EXECUTOR = AutoresearchExecutor()
    return _DEFAULT_EXECUTOR


def execute_autoresearch(
    loop_name: str,
    options: Dict[str, Any],
    *,
    context: Optional[ExecutionContext] = None,
) -> ExecutionResult:
    """Execute an autoresearch loop through the default executor."""
    return get_executor().execute(loop_name, options, context=context)
