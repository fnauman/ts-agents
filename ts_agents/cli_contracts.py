"""Small shared helpers for CLI discovery and contract surfaces."""

from __future__ import annotations


def normalize_cli_template(command: str) -> str:
    """Normalize command templates to install-agnostic `ts-agents ...` forms."""
    normalized = " ".join(command.strip().split())
    if normalized.startswith("uv run ts-agents "):
        return "ts-agents " + normalized[len("uv run ts-agents ") :]
    if normalized.startswith("python -m ts_agents "):
        return "ts-agents " + normalized[len("python -m ts_agents ") :]
    return normalized
