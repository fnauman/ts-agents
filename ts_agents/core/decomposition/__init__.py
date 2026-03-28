"""Time series decomposition methods."""

from __future__ import annotations

from ts_agents._lazy import load_export

_LAZY_EXPORTS = {
    "stl_decompose": ("stl", "stl_decompose"),
    "mstl_decompose": ("mstl", "mstl_decompose"),
    "holt_winters_decompose": ("holt_winters", "holt_winters_decompose"),
}


def __getattr__(name: str):
    value = load_export(__name__, _LAZY_EXPORTS, name)
    globals()[name] = value
    return value

__all__ = [
    "stl_decompose",
    "mstl_decompose",
    "holt_winters_decompose",
]
