"""Run catalog: discover, inspect, and garbage-collect run output directories.

Both workflow runs and autoresearch runs write a ``run_manifest.json`` into
their output directory. This module scans an outputs root for those manifests
and normalizes the two manifest shapes (workflow manifests carry ``workflow``
and no ``kind``; autoresearch manifests carry ``kind == "autoresearch_run"``
and ``loop``) into one catalog record.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from ts_agents.tools.executor import ToolError, ToolErrorCode
from ts_agents.workflows.common import WORKFLOW_MANIFEST_FILENAME

DEFAULT_OUTPUT_ROOT = "outputs"

RUN_KIND_WORKFLOW = "workflow"
RUN_KIND_AUTORESEARCH = "autoresearch"
RUN_KIND_UNKNOWN = "unknown"


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _manifest_created_at(manifest: Dict[str, Any], manifest_path: Path) -> datetime:
    created_at = _parse_timestamp(manifest.get("created_at"))
    if created_at is not None:
        return created_at
    return datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc)


def _normalize_record(manifest: Dict[str, Any], manifest_path: Path) -> Dict[str, Any]:
    if manifest.get("kind") == "autoresearch_run":
        kind = RUN_KIND_AUTORESEARCH
        name = manifest.get("loop")
    elif isinstance(manifest.get("workflow"), str):
        kind = RUN_KIND_WORKFLOW
        name = manifest.get("workflow")
    else:
        kind = RUN_KIND_UNKNOWN
        name = manifest.get("workflow") or manifest.get("loop")

    created_at = _manifest_created_at(manifest, manifest_path)
    return {
        "run_id": manifest.get("run_id"),
        "kind": kind,
        "name": name,
        "status": manifest.get("status"),
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "summary": manifest.get("summary"),
        "output_dir": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
    }


def scan_runs(root: str | Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Scan ``root`` recursively for run manifests.

    Returns normalized records sorted newest-first plus warnings for
    manifests that could not be parsed.
    """
    root_path = Path(root)
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if not root_path.exists():
        return records, warnings

    for manifest_path in sorted(root_path.rglob(WORKFLOW_MANIFEST_FILENAME)):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Skipped unreadable manifest {manifest_path}: {exc}")
            continue
        if not isinstance(manifest, dict):
            warnings.append(
                f"Skipped manifest {manifest_path}: expected a JSON object, "
                f"got {type(manifest).__name__}"
            )
            continue
        records.append(_normalize_record(manifest, manifest_path))

    records.sort(key=lambda record: record["created_at"], reverse=True)
    return records, warnings


def filter_runs(
    records: List[Dict[str, Any]],
    *,
    kind: Optional[str] = None,
    name: Optional[str] = None,
    status: Optional[List[str]] = None,
    older_than_days: Optional[float] = None,
) -> List[Dict[str, Any]]:
    filtered = records
    if kind:
        filtered = [record for record in filtered if record["kind"] == kind]
    if name:
        filtered = [record for record in filtered if record["name"] == name]
    if status:
        wanted = set(status)
        filtered = [record for record in filtered if record["status"] in wanted]
    if older_than_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

        def _is_older(record: Dict[str, Any]) -> bool:
            created = _parse_timestamp(record["created_at"])
            return created is not None and created < cutoff

        filtered = [record for record in filtered if _is_older(record)]
    return filtered


def find_run(root: str | Path, run_id: str) -> Dict[str, Any]:
    """Locate a run by exact id, or by unique id prefix."""
    records, _ = scan_runs(root)
    exact = [record for record in records if record["run_id"] == run_id]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ToolError(
            code=ToolErrorCode.DATA_ERROR,
            message=f"Multiple runs share run_id {run_id!r} under {root}.",
            hint="Pass --root pointing at a single runs tree, or gc duplicates.",
        )

    prefixed = [
        record
        for record in records
        if isinstance(record["run_id"], str) and record["run_id"].startswith(run_id)
    ]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        candidates = ", ".join(sorted(record["run_id"] for record in prefixed)[:5])
        raise ToolError(
            code=ToolErrorCode.VALIDATION_ERROR,
            message=f"Run id prefix {run_id!r} is ambiguous ({candidates}).",
            hint="Use a longer prefix or the full run id.",
        )
    raise ToolError(
        code=ToolErrorCode.NOT_FOUND,
        message=f"No run with id {run_id!r} found under {Path(root).resolve()}.",
        hint="Use `ts-agents runs list` to see known runs.",
    )


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def show_run(root: str | Path, run_id: str) -> Dict[str, Any]:
    record = find_run(root, run_id)
    manifest_path = Path(record["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    artifacts: List[Dict[str, Any]] = []
    for artifact in manifest.get("artifacts", []) or []:
        if not isinstance(artifact, dict):
            continue
        raw_path = artifact.get("path")
        entry = dict(artifact)
        if isinstance(raw_path, str):
            artifact_path = Path(raw_path)
            if not artifact_path.is_absolute():
                artifact_path = manifest_path.parent / artifact_path
            entry["exists"] = artifact_path.exists()
        artifacts.append(entry)

    output_dir = manifest_path.parent
    return {
        **record,
        "size_bytes": _directory_size_bytes(output_dir),
        "artifacts": artifacts,
        "manifest": manifest,
    }


def _safe_gc_target(root_path: Path, output_dir: Path) -> Optional[str]:
    """Return a reason the directory must not be deleted, or None if safe."""
    if output_dir.is_symlink():
        return "output directory is a symlink"
    resolved_root = root_path.resolve()
    resolved_dir = output_dir.resolve()
    if resolved_dir == resolved_root:
        return "output directory is the outputs root itself"
    if resolved_root not in resolved_dir.parents:
        return "output directory escapes the outputs root"
    if not (output_dir / WORKFLOW_MANIFEST_FILENAME).is_file():
        return "output directory no longer contains a run manifest"
    return None


def gc_runs(
    root: str | Path,
    *,
    kind: Optional[str] = None,
    name: Optional[str] = None,
    status: Optional[List[str]] = None,
    older_than_days: Optional[float] = None,
    apply: bool = False,
) -> Dict[str, Any]:
    """Delete (or preview deleting) run directories matching the filters.

    Without ``apply`` this is a dry run listing candidates. Nested runs are
    handled by deleting outermost directories first and skipping children
    that disappear with their parent.
    """
    root_path = Path(root)
    records, warnings = scan_runs(root_path)
    candidates = filter_runs(
        records,
        kind=kind,
        name=name,
        status=status,
        older_than_days=older_than_days,
    )

    deleted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    freed_bytes = 0
    for record in sorted(candidates, key=lambda item: len(Path(item["output_dir"]).parts)):
        output_dir = Path(record["output_dir"])
        if not output_dir.exists():
            continue
        reason = _safe_gc_target(root_path, output_dir)
        if reason is not None:
            skipped.append({**record, "reason": reason})
            continue
        size_bytes = _directory_size_bytes(output_dir)
        freed_bytes += size_bytes
        if apply:
            shutil.rmtree(output_dir)
        deleted.append({**record, "size_bytes": size_bytes})

    return {
        "root": str(root_path.resolve()),
        "dry_run": not apply,
        "matched": len(deleted),
        "freed_bytes": freed_bytes,
        "runs": deleted,
        "skipped": skipped,
        "warnings": warnings,
    }


def render_runs_table(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "No runs found."
    headers = ("RUN_ID", "KIND", "NAME", "STATUS", "CREATED_AT")
    rows = [
        (
            str(record.get("run_id") or "-"),
            record["kind"],
            str(record.get("name") or "-"),
            str(record.get("status") or "-"),
            record["created_at"],
        )
        for record in records
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
    ]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)
