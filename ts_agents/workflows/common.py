"""Shared helpers for workflow artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ts_agents.cli.output import render_output, to_jsonable, write_output
from ts_agents.contracts import ArtifactRef


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        path=str(Path(path)),
        mime_type=mime_type,
        description=description,
        created_by=created_by,
    )
