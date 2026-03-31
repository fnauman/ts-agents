"""Shared machine-readable contracts for CLI and tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ArtifactRef:
    """Structured description of an output artifact."""

    kind: str
    path: str
    mime_type: Optional[str] = None
    description: Optional[str] = None
    created_by: Optional[str] = None


@dataclass
class ToolPayload:
    """Structured payload returned by machine-oriented tool wrappers."""

    kind: str
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactRef] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CLIExecution:
    """Execution metadata included in CLI envelopes."""

    backend_requested: Optional[str] = None
    backend_actual: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CLIError:
    """Structured CLI error payload."""

    code: str
    message: str
    retryable: bool = False
    hint: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CLIEnvelope:
    """Top-level CLI success/error envelope."""

    ok: bool
    command: str
    name: Optional[str] = None
    input: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[CLIError] = None
    execution: Optional[CLIExecution] = None
