"""Helpers for package-level lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Mapping


LazyExportMap = Mapping[str, tuple[str, str]]


def load_export(package_name: str, exports: LazyExportMap, name: str):
    """Import and return a lazily exported attribute."""
    try:
        module_name, attr_name = exports[name]
    except KeyError as exc:  # pragma: no cover - tiny helper
        raise AttributeError(
            f"module {package_name!r} has no attribute {name!r}"
        ) from exc

    module = import_module(f"{package_name}.{module_name}")
    value = getattr(module, attr_name)
    return value
