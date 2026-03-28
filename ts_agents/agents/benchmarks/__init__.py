"""Benchmarking infrastructure for agent performance analysis."""

from __future__ import annotations

from ts_agents._lazy import load_export

_LAZY_EXPORTS = {
    "AgentBenchmark": ("runner", "AgentBenchmark"),
    "BenchmarkResult": ("runner", "BenchmarkResult"),
    "BENCHMARK_SCENARIOS": ("scenarios", "BENCHMARK_SCENARIOS"),
    "BenchmarkScenario": ("scenarios", "BenchmarkScenario"),
    "compute_agent_metrics": ("metrics", "compute_agent_metrics"),
    "evaluate_response": ("metrics", "evaluate_response"),
}


def __getattr__(name: str):
    value = load_export(__name__, _LAZY_EXPORTS, name)
    globals()[name] = value
    return value

__all__ = [
    "AgentBenchmark",
    "BenchmarkResult",
    "BENCHMARK_SCENARIOS",
    "BenchmarkScenario",
    "compute_agent_metrics",
    "evaluate_response",
]
