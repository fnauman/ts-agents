"""Command-line interface for ts-agents."""

from __future__ import annotations

from typing import Any


def main(*args: Any, **kwargs: Any) -> Any:
    """Lazily load the CLI entrypoint to avoid package import cycles."""
    from .main import main as _main

    return _main(*args, **kwargs)


__all__ = ["main"]
