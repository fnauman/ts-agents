"""Compatibility checks for rerunning a workflow in an existing directory."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import tempfile
from datetime import datetime, timezone
from typing import Any

from ts_agents.cli.output import to_jsonable


def _identity_value(value: Any) -> Any:
    """Encode input types/values without presentation-time JSON sanitization."""
    import numpy as np

    if isinstance(value, np.ndarray):
        content = (_identity_value(value.tolist()) if value.dtype.hasobject
                   else hashlib.sha256(value.tobytes(order="C")).hexdigest())
        return ["array", str(value.dtype), list(value.shape), content]
    if isinstance(value, np.generic):
        return _identity_value(value.item())
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, dict):
        pairs = [[_identity_value(key), _identity_value(item)] for key, item in value.items()]
        return ["dict", sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_identity_value(item) for item in value]]
    return [type(value).__module__, type(value).__name__, to_jsonable(value)]


def _identity_digest(value: Any) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":"))
    for chunk in encoder.iterencode(_identity_value(value)):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def resume_identity(workflow: str, workflow_input: Any, options: dict) -> dict:
    """Fingerprint all normalized input fields and analysis/presentation options.

    Only output location is excluded. Resume is a rerun, not checkpoint recovery.
    Numeric/nonfinite distinctions survive in arrays, labels, times, and metadata.
    """
    fields = (
        "series", "values", "labels", "time_values", "time_column",
        "value_column", "value_columns", "label_column", "source_type",
        "input_path", "label", "provenance",
    )
    normalized = {name: getattr(workflow_input, name) for name in fields if hasattr(workflow_input, name)}
    analysis_options = {key: value for key, value in options.items() if key != "output_dir"}
    return {
        "version": "1", "workflow": workflow,
        "input_sha256": _identity_digest(normalized),
        "options_sha256": _identity_digest(analysis_options),
        "options": to_jsonable(analysis_options),
    }


def validate_resume(manifest: dict, identity: dict) -> None:
    """Reject incompatible or unverifiable reruns before changing any files."""
    if manifest.get("workflow") != identity["workflow"]:
        raise ValueError("--resume requires the same workflow as the existing run.")
    if manifest.get("status") == "running":
        raise ValueError("--resume refuses a running or interrupted run; inspect it and use a new --output-dir.")
    previous = manifest.get("resume_identity")
    if not isinstance(previous, dict):
        raise ValueError(
            "--resume cannot verify this legacy run's input and options. "
            "Use a new --output-dir to preserve the existing evidence."
        )
    if previous != identity:
        raise ValueError(
            "--resume requires identical input content, source interpretation, and options. "
            "Use a new --output-dir for a different analysis."
        )


def write_manifest(path: str | Path, manifest: dict) -> None:
    """Replace a manifest atomically; never advertise success on write failure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(to_jsonable(manifest), handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def lock_run_output(output_dir: str | Path):
    """Hold shared ancestor leases and an exclusive output lease.

    A parent overwrite/GC therefore conflicts with every nested writer,
    including one that has not published a manifest. Independent sibling runs
    can execute concurrently on POSIX. Leases are outside deletable run trees.
    Native Windows uses conservative exclusive locks for every scope.
    """
    from contextlib import ExitStack
    from ts_agents.cli.jobs import _try_lock_lease, _unlock_lease

    output = Path(output_dir).resolve()
    user = str(os.getuid()) if hasattr(os, "getuid") else os.environ.get("USERNAME", "default")
    lock_root = Path(tempfile.gettempdir()) / ("ts-agents-run-locks-" + hashlib.sha256(user.encode()).hexdigest()[:16])
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    scopes = [*reversed(output.parents), output]
    with ExitStack() as stack:
        directory_fd = None
        if os.name == "posix":
            directory_fd = os.open(lock_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, directory_fd)
            root_stat = os.fstat(directory_fd)
            if root_stat.st_uid != os.getuid() or root_stat.st_mode & 0o077:
                raise ValueError("Run lease directory must be owned by the current user and private.")
        elif lock_root.is_symlink():
            raise ValueError("Run lease directory must not be a symlink.")
        for scope in scopes:
            name = hashlib.sha256(str(scope).encode()).hexdigest()
            if directory_fd is not None:
                fd = os.open(f"{name}.lease", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                             0o600, dir_fd=directory_fd)
                try:
                    lease_stat = os.fstat(fd)
                    if (not stat.S_ISREG(lease_stat.st_mode) or lease_stat.st_uid != os.getuid()
                            or lease_stat.st_nlink != 1):
                        raise ValueError("Run lease file must be a private, owned regular file with no links.")
                    os.fchmod(fd, 0o600)
                    handle = os.fdopen(fd, "a+b")
                except BaseException:
                    os.close(fd)
                    raise
                stack.enter_context(handle)
            else:
                lease_path = lock_root / f"{name}.lease"
                if lease_path.is_symlink():
                    raise ValueError("Run lease file must not be a symlink.")
                handle = stack.enter_context(lease_path.open("a+b"))
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            if os.name == "posix":
                import fcntl
                operation = fcntl.LOCK_EX if scope == output else fcntl.LOCK_SH
                try:
                    fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError(f"Run output directory is busy: {output}") from exc
            elif not _try_lock_lease(handle):
                raise ValueError(f"Run output directory is busy: {output}")
            stack.callback(_unlock_lease, handle)
        yield


def begin_run(lifecycle: dict, identity: dict, source: dict) -> str:
    """Catalog active work before execution so GC can retain it."""
    previous = lifecycle.get("_existing_manifest") or {}
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": "1.0", "workflow": identity["workflow"],
        "run_id": lifecycle["run_id"], "status": "running",
        "created_at": previous.get("created_at", now), "updated_at": now,
        "output_dir": lifecycle["output_dir"], "manifest_path": lifecycle["manifest_path"],
        "source": source, "options": identity["options"], "resume_identity": identity,
        "artifacts": [],
    }
    write_manifest(lifecycle["manifest_path"], manifest)
    lifecycle["_created_at"] = manifest["created_at"]
    return manifest["created_at"]


def fail_run(lifecycle: dict, error: Exception, identity: dict) -> None:
    """Retain failed-run identity and the error for catalog inspection."""
    path = Path(lifecycle["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["status"] = "failed"
    manifest["created_at"] = lifecycle["_created_at"]
    manifest["resume_identity"] = identity
    manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    write_manifest(path, manifest)
