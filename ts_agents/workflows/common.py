"""Shared helpers for workflow artifact generation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any
import uuid

from ts_agents.cli.output import render_output, to_jsonable, write_output
from ts_agents.contracts import ArtifactRef, CLI_SCHEMA_VERSION, ToolPayload

WORKFLOW_MANIFEST_FILENAME = "run_manifest.json"


def generate_workflow_run_id() -> str:
    """Create a stable, time-ordered workflow run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir_has_files(output_dir: str | Path) -> bool:
    path = Path(output_dir)
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return any(path.iterdir())


def clear_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).resolve()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_workflow_manifest(output_dir: str | Path) -> dict[str, Any] | None:
    manifest_path = Path(output_dir) / WORKFLOW_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{WORKFLOW_MANIFEST_FILENAME} in {Path(output_dir).resolve()} is not valid JSON "
            f"and cannot be used for --resume. ({exc})"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"{WORKFLOW_MANIFEST_FILENAME} in {Path(output_dir).resolve()} must contain a JSON object "
            f"for --resume, got {type(payload).__name__}."
        )

    return payload


def write_json_artifact(
    *,
    data: Any,
    path: str | Path,
    description: str,
    created_by: str,
) -> ArtifactRef:
    output_path = Path(path)
    payload = render_output(to_jsonable(data), json_output=True)
    write_output(payload, str(output_path))
    return artifact_ref(
        kind="json",
        path=output_path,
        mime_type="application/json",
        description=description,
        created_by=created_by,
    )


def write_text_artifact(
    *,
    content: str,
    path: str | Path,
    description: str,
    created_by: str,
    mime_type: str = "text/markdown",
) -> ArtifactRef:
    output_path = Path(path)
    write_output(content, str(output_path))
    return artifact_ref(
        kind="markdown" if mime_type == "text/markdown" else "text",
        path=output_path,
        mime_type=mime_type,
        description=description,
        created_by=created_by,
    )


def write_dataframe_artifact(
    *,
    dataframe,
    path: str | Path,
    description: str,
    created_by: str,
) -> ArtifactRef:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return artifact_ref(
        kind="csv",
        path=output_path,
        mime_type="text/csv",
        description=description,
        created_by=created_by,
    )


def write_plot_artifact(
    *,
    figure,
    path: str | Path,
    description: str,
    created_by: str,
) -> ArtifactRef:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png")
    return artifact_ref(
        kind="image",
        path=output_path,
        mime_type="image/png",
        description=description,
        created_by=created_by,
    )


def artifact_ref(
    *,
    kind: str,
    path: str | Path,
    mime_type: str,
    description: str,
    created_by: str,
) -> ArtifactRef:
    return ArtifactRef(
        kind=kind,
        path=str(Path(path).resolve()),
        mime_type=mime_type,
        description=description,
        created_by=created_by,
    )


def attach_workflow_run_metadata(
    payload: ToolPayload,
    *,
    workflow_name: str,
    output_dir: str | Path,
    run_id: str | None,
    source: Any,
    options: dict[str, Any],
    resumed: bool = False,
    output_dir_mode: str = "explicit",
) -> ToolPayload:
    """Attach run metadata to a workflow payload and write a manifest artifact."""
    output_path = Path(output_dir).resolve()
    resolved_run_id = run_id or generate_workflow_run_id()
    manifest_path = output_path / WORKFLOW_MANIFEST_FILENAME
    run_metadata = {
        "run_id": resolved_run_id,
        "output_dir": str(output_path),
        "manifest_path": str(manifest_path),
        "manifest_filename": WORKFLOW_MANIFEST_FILENAME,
        "resumed": bool(resumed),
        "output_dir_mode": output_dir_mode,
    }

    if not isinstance(payload.data, dict):
        raise TypeError(
            f"attach_workflow_run_metadata: payload.data must be a dict, got {type(payload.data).__name__}"
        )
    payload.data.update(run_metadata)
    payload.data["run"] = dict(run_metadata)

    manifest_artifact = artifact_ref(
        kind="json",
        path=manifest_path,
        mime_type="application/json",
        description="Workflow run manifest.",
        created_by=workflow_name,
    )

    manifest_payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "workflow": workflow_name,
        "run_id": resolved_run_id,
        "status": payload.status,
        "summary": payload.summary,
        "output_dir": str(output_path),
        "manifest_path": str(manifest_path),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "resumed": bool(resumed),
        "output_dir_mode": output_dir_mode,
        "source": to_jsonable(source),
        "options": to_jsonable(options),
        "warnings": to_jsonable(payload.warnings),
        "quality_flags": to_jsonable(payload.data.get("quality_flags") or []),
        "artifacts": [to_jsonable(artifact) for artifact in (*payload.artifacts, manifest_artifact)],
        "provenance": to_jsonable(payload.provenance),
    }
    payload_text = render_output(to_jsonable(manifest_payload), json_output=True)
    write_output(payload_text, str(manifest_path))
    payload.artifacts.append(manifest_artifact)
    return payload
