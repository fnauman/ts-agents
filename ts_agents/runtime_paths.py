"""Resolve runtime paths for repo checkouts and packaged installs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

_MODULE_ROOT = Path(__file__).resolve().parent
_CHECKOUT_ROOT = _MODULE_ROOT.parent
_RESOURCES_ROOT = _MODULE_ROOT / "resources"


def resolve_existing_path(path: PathLike) -> Optional[Path]:
    """Resolve an existing path in checkout root or packaged resources.

    Resolution order:
    1) Absolute path as provided.
    2) Relative to repository checkout root.
    3) Relative to package resource root (ts_agents/resources/).
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    checkout_candidate = _CHECKOUT_ROOT / candidate
    if checkout_candidate.exists():
        return checkout_candidate

    resource_candidate = _RESOURCES_ROOT / candidate
    if resource_candidate.exists():
        return resource_candidate

    return None


def resolve_default_data_dir() -> Path:
    """Resolve default data directory independent of current working directory."""
    resolved = resolve_existing_path("data")
    if resolved and resolved.is_dir():
        return resolved
    return Path.cwd() / "data"


def resolve_default_skills_dir() -> Path:
    """Resolve default skills directory independent of current working directory."""
    resolved = resolve_existing_path("skills")
    if resolved and resolved.is_dir():
        return resolved
    return Path.cwd() / "skills"
