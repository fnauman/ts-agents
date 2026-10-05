"""Compatibility checks for rerunning a workflow in an existing directory."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from typing import Any

from ts_agents.cli.output import to_jsonable


def resume_identity(workflow: str, workflow_input: Any, options: dict) -> dict:
    """Fingerprint normalized input, interpretation, and options.

    Only output location is excluded. Resume requires the same analysis and
    presentation settings; it is a rerun, not checkpoint recovery.
    """
    fields = (
        "series", "values", "labels", "time_values", "time_column",
        "value_column", "value_columns", "label_column", "source_type",
        "input_path", "label", "provenance",
    )
    normalized = {
        name: to_jsonable(getattr(workflow_input, name))
        for name in fields if hasattr(workflow_input, name)
    }
    # Preserve nonfinite values distinctly: JSON sanitization would otherwise
    # collapse NaN and both infinities to null in the compatibility digest.
    for name in ("series", "values"):
        array = getattr(workflow_input, name, None)
        if array is not None:
            normalized[name] = {
                "dtype": str(array.dtype), "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":"))
    for chunk in encoder.iterencode(normalized):
        digest.update(chunk.encode("utf-8"))
    return {
        "version": "1",
        "workflow": workflow,
        "input_sha256": digest.hexdigest(),
        "options": to_jsonable({key: value for key, value in options.items() if key != "output_dir"}),
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
    """Exclude simultaneous writers/cleanup of the same output directory.

    Locks live outside the directory so overwrite/GC cannot unlink an active
    lease and let another process acquire a different inode for the same run.
    """
    from ts_agents.cli.jobs import _try_lock_lease, _unlock_lease

    output = Path(output_dir).resolve()
    lock_root = output.parent / ".ts-agents-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(str(output).encode()).hexdigest()
    with (lock_root / f"{name}.lease").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        if not _try_lock_lease(handle):
            raise ValueError(f"Run output directory is busy: {output}")
        try:
            yield
        finally:
            _unlock_lease(handle)


def begin_run(lifecycle: dict, identity: dict, source: dict) -> None:
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


def fail_run(lifecycle: dict, error: Exception, identity: dict) -> None:
    """Retain failed-run identity and the error for catalog inspection."""
    path = Path(lifecycle["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["status"] = "failed"
    manifest["resume_identity"] = identity
    manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    write_manifest(path, manifest)
