"""Sandbox-aware execution for autoresearch loops."""

from __future__ import annotations

import base64
import binascii
from dataclasses import replace
from importlib.util import find_spec
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Tuple

from ts_agents.cli.output import to_jsonable
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

from .registry import get_loop
from .runner import run_autoresearch_loop

_AUTORESEARCH_PREFIX = "autoresearch:"
_SANDBOX_ARTIFACT_DIR_ENV = _staging.SANDBOX_ARTIFACT_DIR_ENV
_STAGED_AUTORESEARCH_ARTIFACTS_KEY = "_ts_agents_staged_autoresearch_artifacts"
_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES_ENV = _staging.AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES_ENV
_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES_ENV = _staging.AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES_ENV
_WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV = _staging.WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV
_WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV = _staging.WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV

_enforce_host_availability_for_backend = _staging.enforce_host_availability_for_backend
_append_payload_warning = _staging.append_payload_warning
_path_contains_symlink = _staging.path_contains_symlink
_valid_relative_artifact_path = _staging.valid_relative_artifact_path
_safe_destination_for_relative_path = _staging.safe_destination_for_relative_path
_relative_artifact_path = _staging.relative_artifact_path
_write_bytes_atomically = _staging.write_bytes_atomically


def _autoresearch_target_name(loop_name: str) -> str:
    return f"{_AUTORESEARCH_PREFIX}{loop_name}"


def is_autoresearch_target(tool_name: str) -> bool:
    """Return whether a sandbox request targets an autoresearch loop."""
    return tool_name.startswith(_AUTORESEARCH_PREFIX)


def _autoresearch_artifact_bundle_limits() -> Tuple[Optional[int], Optional[int]]:
    # Workflow limit env vars still apply as a fallback so one knob can tune
    # both staging surfaces.
    return _staging.artifact_bundle_limits(
        (_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES_ENV, _WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV),
        (_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES_ENV, _WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV),
    )


def _selected_models_for_availability(
    loop_definition: Any, options: Dict[str, Any]
) -> list[str]:
    models = options.get("models")
    if models is None:
        return list(loop_definition.models)
    return [
        str(model).strip()
        for model in models
        if model is not None and str(model).strip()
    ]


def _autoresearch_availability(
    loop_definition: Any, options: Dict[str, Any]
) -> Dict[str, Any]:
    """Check the loop's declared dependency rules against the host."""
    models = _selected_models_for_availability(loop_definition, options)
    missing: list[str] = []
    hint_label = "dependencies"
    hint_extra: Optional[str] = None
    for rule in getattr(loop_definition, "dependency_rules", ()) or ():
        if rule.models is not None and not set(models).intersection(rule.models):
            continue
        if rule.skip_on_dry_run and options.get("dry_run"):
            continue
        rule_missing = [
            distribution_name
            for import_name, distribution_name in rule.modules
            if find_spec(import_name) is None
        ]
        if rule_missing:
            missing.extend(rule_missing)
            if hint_extra is None:
                hint_label = rule.label
                hint_extra = rule.install_extra
    if not missing:
        return {"available": True, "missing": []}
    install_hint = (
        f"Autoresearch loop {loop_definition.name!r} requires optional "
        f"{hint_label}: {', '.join(missing)}. Install with: "
        f"pip install 'ts-agents[{hint_extra}]'"
    )
    return {
        "available": False,
        "missing": missing,
        "install_hint": install_hint,
    }


def _context_with_loop_environment(
    context: ExecutionContext,
    loop_definition: Any,
    actual_backend: SandboxMode,
) -> ExecutionContext:
    if actual_backend != SandboxMode.DAYTONA or not loop_definition.required_extras:
        return context
    env = dict(context.environment or {})
    env["TS_AGENTS_DAYTONA_INSTALL_EXTRAS"] = ",".join(loop_definition.required_extras)
    return replace(context, environment=env)


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
    staged_files = _staging.collect_staged_artifact_files(
        output_dir,
        payload,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        file_limit_env=_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES_ENV,
        total_limit_env=_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES_ENV,
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

        if (
            backend is None
            or not requested_status["available"]
            or not backend.is_available()
        ):
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
            if (
                backend is None
                or not fallback_status["available"]
                or not backend.is_available()
            ):
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=ToolError(
                        code=ToolErrorCode.BACKEND_UNAVAILABLE,
                        message=(
                            f"Requested backend '{requested_backend.value}' is unavailable and fallback backend "
                            f"'{fallback_backend.value}' is also unavailable."
                        ),
                        recoverable=True,
                        hint=fallback_status.get("suggested_fix")
                        or requested_status.get("suggested_fix"),
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

        availability = _autoresearch_availability(loop_definition, dict(options or {}))
        if _enforce_host_availability_for_backend(
            actual_backend
        ) and not availability.get("available", True):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=ToolError(
                    code=ToolErrorCode.DEPENDENCY_ERROR,
                    message=availability.get("install_hint")
                    or f"Autoresearch loop '{loop_name}' is unavailable in the current environment.",
                    recoverable=False,
                    tool_name=loop_name,
                    details={"availability": availability},
                ),
                metadata={
                    "loop_name": loop_name,
                    "backend_requested": requested_backend.value,
                    "backend_actual": actual_backend.value,
                    "fallback_allowed": context.allow_fallback,
                    "fallback_backend": fallback_backend.value
                    if context.allow_fallback
                    else None,
                    "fallback_used": actual_backend != requested_backend,
                    "availability": availability,
                },
            )

        execution_context = _context_with_loop_environment(
            context, loop_definition, actual_backend
        )

        requested_output_dir = dict(options or {}).get("output_dir")
        request_payload = {
            "loop_name": loop_name,
            "options": dict(options or {}),
            "use_sandbox_artifact_dir": _use_staged_autoresearch_artifact_dir(
                actual_backend
            ),
            "sandbox_artifact_dir": _sandbox_autoresearch_artifact_dir(actual_backend),
            "bundle_sandbox_artifacts": _bundle_staged_autoresearch_artifacts(
                actual_backend
            ),
        }

        result = backend.execute(
            tool_name=_autoresearch_target_name(loop_name),
            func=_run_serialized_autoresearch,
            params=request_payload,
            context=execution_context,
        )
        result.metadata = {
            **(result.metadata or {}),
            "loop_name": loop_name,
            "backend_requested": requested_backend.value,
            "backend_actual": actual_backend.value,
            "fallback_used": actual_backend != requested_backend,
            "fallback_allowed": context.allow_fallback,
            "fallback_backend": fallback_backend.value
            if context.allow_fallback
            else None,
        }

        if (
            result.success
            and actual_backend == SandboxMode.DOCKER
            and requested_output_dir
        ):
            _materialize_existing_artifacts(result, requested_output_dir)
        elif result.success and actual_backend in {
            SandboxMode.DAYTONA,
            SandboxMode.MODAL,
        }:
            _materialize_remote_autoresearch_output(result, requested_output_dir)
        return result


def _use_staged_autoresearch_artifact_dir(backend: SandboxMode) -> bool:
    return _staging.use_staged_artifact_dir(backend)


def _bundle_staged_autoresearch_artifacts(backend: SandboxMode) -> bool:
    return _staging.bundle_staged_artifacts(backend)


def _sandbox_autoresearch_artifact_dir(backend: SandboxMode) -> Optional[str]:
    return _staging.sandbox_artifact_dir(
        backend, modal_prefix="ts_agents_autoresearch_artifacts"
    )


def _materialize_existing_artifacts(
    result: ExecutionResult, requested_output_dir: str
) -> None:
    payload = result.result
    if not isinstance(payload, dict):
        return
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return

    destination_dir = Path(requested_output_dir).expanduser().absolute()
    if _path_contains_symlink(destination_dir):
        _append_payload_warning(
            payload,
            f"Skipped restoring Docker artifacts because output directory '{destination_dir}' contains a symlink component.",
        )
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    source_root_raw = data.get("output_dir") if isinstance(data, dict) else None
    source_root = (
        Path(source_root_raw).resolve() if isinstance(source_root_raw, str) else None
    )
    rewritten: dict[str, Path] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        source_path = artifact.get("path")
        if not isinstance(source_path, str):
            continue
        source = Path(source_path)
        if not source.exists() or not source.is_file():
            continue
        relative_path = _relative_artifact_path(source, source_root)
        destination = _safe_destination_for_relative_path(
            destination_dir, relative_path, payload
        )
        if destination is None:
            continue
        if source.resolve() != destination.resolve(strict=False):
            try:
                _write_bytes_atomically(destination, source.read_bytes())
            except OSError as exc:
                _append_payload_warning(
                    payload,
                    f"Skipped restoring Docker artifact '{relative_path}' because it could not be written: {exc}",
                )
                continue
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
        Path(requested_output_dir).expanduser().absolute()
        if requested_output_dir
        else Path(tempfile.mkdtemp(prefix="ts_agents_autoresearch_output_")).absolute()
    )
    if _path_contains_symlink(destination_root):
        _append_payload_warning(
            payload,
            f"Skipped restoring remote artifacts because output directory '{destination_root}' contains a symlink component.",
        )
        return
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
        destination = _safe_destination_for_relative_path(
            destination_root, relative_path, payload
        )
        if destination is None:
            continue
        try:
            content = base64.b64decode(content_base64.encode("ascii"))
        except (ValueError, binascii.Error):
            _append_payload_warning(
                payload,
                f"Skipped restoring remote artifact '{relative_path}' because its payload was not valid base64.",
            )
            continue
        try:
            _write_bytes_atomically(destination, content)
        except OSError as exc:
            _append_payload_warning(
                payload,
                f"Skipped restoring remote artifact '{relative_path}' because it could not be written: {exc}",
            )
            continue
        if isinstance(source_path, str):
            rewritten[source_path] = destination

    _rewrite_remote_artifact_refs(payload, rewritten)
    _rewrite_payload_paths(payload, destination_root, rewritten)
    result.formatted_output = format_result(payload)


def _rewrite_remote_artifact_refs(
    payload: Dict[str, Any], rewritten: dict[str, Path]
) -> None:
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
