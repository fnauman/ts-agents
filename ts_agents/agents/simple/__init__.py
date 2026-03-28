"""Simple LangChain-based agent for testing tool scaling."""

from __future__ import annotations

from ts_agents._lazy import load_export

_LAZY_EXPORTS = {
    "create_simple_agent": ("agent", "create_simple_agent"),
    "SimpleAgentChat": ("agent", "SimpleAgentChat"),
}


def __getattr__(name: str):
    value = load_export(__name__, _LAZY_EXPORTS, name)
    globals()[name] = value
    return value

__all__ = [
    "create_simple_agent",
    "SimpleAgentChat",
]
