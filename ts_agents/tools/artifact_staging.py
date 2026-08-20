"""Shared artifact-staging primitives for sandbox executors.

The autoresearch and workflow executors both move artifacts produced inside a
sandbox back onto the host. The path-validation, symlink-rejection, and
atomic-write rules live here so the two surfaces cannot drift apart again;
each executor keeps only its payload-shape-specific logic.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from ts_agents.tools.executor import SandboxMode

SANDBOX_ARTIFACT_DIR_ENV = "TS_AGENTS_TOOL_ARTIFACT_DIR"

WORKFLOW_ARTIFACT_MAX_FILE_BYTES_ENV = "TS_AGENTS_WORKFLOW_ARTIFACT_MAX_FILE_BYTES"
WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES_ENV = "TS_AGENTS_WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES"
AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES_ENV = (
    "TS_AGENTS_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES"
)
AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES_ENV = (
    "TS_AGENTS_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES"
)

DEFAULT_ARTIFACT_MAX_FILE_BYTES = 16 * 1024 * 1024
DEFAULT_ARTIFACT_MAX_TOTAL_BYTES = 64 * 1024 * 1024


def parse_artifact_limit(var_names: Sequence[str], default: int) -> Optional[int]:
    """Read the first parseable limit from ``var_names``; <=0 disables the limit."""
    for var_name in var_names:
        raw = os.environ.get(var_name)
        if raw is None or not raw.strip():
            continue
        try:
            value = int(raw.strip())
        except ValueError:
            continue
        if value <= 0:
            return None
        return value
    return default


def enforce_host_availability_for_backend(backend: SandboxMode) -> bool:
    """Whether host dependency availability gates execution for this backend."""
    return backend in {SandboxMode.LOCAL, SandboxMode.SUBPROCESS}


def use_staged_artifact_dir(backend: SandboxMode) -> bool:
    return backend in {SandboxMode.DOCKER, SandboxMode.DAYTONA, SandboxMode.MODAL}


def bundle_staged_artifacts(backend: SandboxMode) -> bool:
    return backend in {SandboxMode.DAYTONA, SandboxMode.MODAL}


def sandbox_artifact_dir(backend: SandboxMode, *, modal_prefix: str) -> Optional[str]:
    """Return a per-run staging directory inside the sandbox.

    Every backend gets a unique suffix so concurrent runs cannot clobber each
    other's staged artifacts.
    """
    if backend == SandboxMode.DOCKER:
        return f"/io/artifacts/{uuid.uuid4().hex[:8]}"
    if backend == SandboxMode.DAYTONA:
        return f".ts_agents_io/artifacts/{uuid.uuid4().hex[:8]}"
    if backend == SandboxMode.MODAL:
        return f"/tmp/{modal_prefix}/{uuid.uuid4().hex[:8]}"
    return None


def append_payload_warning(payload: Dict[str, Any], warning: str) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        payload["warnings"] = warnings
    if warning not in warnings:
        warnings.append(warning)


def path_contains_symlink(path: Path) -> bool:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
        if not current.exists():
            return False
    return False


def valid_relative_artifact_path(relative_path: str) -> Optional[Path]:
    candidate = Path(relative_path)
    if candidate.is_absolute() or not candidate.parts:
        return None
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    return candidate


def safe_destination_for_relative_path(
    destination_root: Path,
    relative_path: str,
    payload: Dict[str, Any],
) -> Optional[Path]:
    candidate = valid_relative_artifact_path(relative_path)
    if candidate is None:
        append_payload_warning(
            payload,
            f"Skipped restoring artifact '{relative_path}' because it is not a safe relative path.",
        )
        return None

    parent = destination_root
    for part in candidate.parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            append_payload_warning(
                payload,
                f"Skipped restoring artifact '{relative_path}' because a destination directory is a symlink.",
            )
            return None
        try:
            parent.mkdir(exist_ok=True)
        except OSError as exc:
            append_payload_warning(
                payload,
                f"Skipped restoring artifact '{relative_path}' because its destination directory could not be created: {exc}",
            )
            return None
        if not parent.is_dir() or parent.is_symlink():
            append_payload_warning(
                payload,
                f"Skipped restoring artifact '{relative_path}' because its destination directory is unsafe.",
            )
            return None

    destination = parent / candidate.name
    if destination.is_symlink():
        append_payload_warning(
            payload,
            f"Skipped restoring artifact '{relative_path}' because the destination is a symlink.",
        )
        return None
    return destination


def relative_artifact_path(source: Path, source_root: Optional[Path]) -> str:
    if source_root is not None:
        try:
            relative = source.resolve().relative_to(source_root)
            safe = valid_relative_artifact_path(relative.as_posix())
            if safe is not None:
                return safe.as_posix()
        except ValueError:
            pass
    return source.name


def write_bytes_atomically(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_bytes(content)
        os.replace(temp_path, destination)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def collect_staged_artifact_files(
    output_dir: Path,
    payload: Dict[str, Any],
    *,
    max_file_bytes: Optional[int],
    max_total_bytes: Optional[int],
    file_limit_env: str,
    total_limit_env: str,
) -> List[Dict[str, str]]:
    """Base64-bundle files under ``output_dir`` for transport back to the host."""
    total_bytes = 0
    staged_files: List[Dict[str, str]] = []
    if not output_dir.exists():
        return staged_files
    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(output_dir).as_posix()
        if file_path.is_symlink() or path_contains_symlink(file_path):
            append_payload_warning(
                payload,
                f"Skipped remote artifact staging for '{relative_path}' because it is or contains a symlink.",
            )
            continue
        try:
            file_size = file_path.stat().st_size
        except OSError as exc:
            append_payload_warning(
                payload,
                f"Skipped remote artifact staging for '{relative_path}' because it could not be inspected: {exc}",
            )
            continue
        if max_file_bytes is not None and file_size > max_file_bytes:
            append_payload_warning(
                payload,
                "Skipped remote artifact staging for "
                f"'{relative_path}' because it exceeds the per-file limit of {max_file_bytes} bytes. "
                f"Set {file_limit_env} to override.",
            )
            continue
        if max_total_bytes is not None and total_bytes + file_size > max_total_bytes:
            append_payload_warning(
                payload,
                "Skipped remote artifact staging for "
                f"'{relative_path}' because bundling it would exceed the total limit of {max_total_bytes} bytes. "
                f"Set {total_limit_env} to override.",
            )
            continue
        try:
            content = file_path.read_bytes()
        except OSError as exc:
            append_payload_warning(
                payload,
                f"Skipped remote artifact staging for '{relative_path}' because it could not be read: {exc}",
            )
            continue
        total_bytes += len(content)
        staged_files.append(
            {
                "source_path": str(file_path.resolve()),
                "relative_path": relative_path,
                "content_base64": base64.b64encode(content).decode("ascii"),
            }
        )
    return staged_files


def artifact_bundle_limits(
    file_env_names: Sequence[str],
    total_env_names: Sequence[str],
) -> Tuple[Optional[int], Optional[int]]:
    return (
        parse_artifact_limit(file_env_names, DEFAULT_ARTIFACT_MAX_FILE_BYTES),
        parse_artifact_limit(total_env_names, DEFAULT_ARTIFACT_MAX_TOTAL_BYTES),
    )
