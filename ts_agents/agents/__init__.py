"""Agent implementations for time series analysis."""

from __future__ import annotations

from ts_agents._lazy import load_export

_LAZY_EXPORTS = {
    "create_simple_agent": ("simple", "create_simple_agent"),
    "SimpleAgentChat": ("simple", "SimpleAgentChat"),
    "create_deep_agent": ("deep", "create_deep_agent"),
    "DeepAgentChat": ("deep", "DeepAgentChat"),
    "list_subagents": ("deep", "list_subagents"),
    "AgentBenchmark": ("benchmarks", "AgentBenchmark"),
    "BenchmarkResult": ("benchmarks", "BenchmarkResult"),
}


def __getattr__(name: str):
    value = load_export(__name__, _LAZY_EXPORTS, name)
    globals()[name] = value
    return value

__all__ = [
    # Simple agent
    "create_simple_agent",
    "SimpleAgentChat",

    # Deep agent
    "create_deep_agent",
    "DeepAgentChat",
    "list_subagents",

    # Benchmarks
    "AgentBenchmark",
    "BenchmarkResult",
]
