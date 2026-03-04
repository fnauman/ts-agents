"""Benchmarking infrastructure for agent performance analysis.

This module provides:
- BenchmarkScenario: Test scenarios with queries and expected outcomes
- AgentBenchmark: Runner for executing benchmarks
- BenchmarkResult: Structured results with metrics

Example usage:
    >>> from ts_agents.agents.benchmarks import AgentBenchmark
    >>>
    >>> benchmark = AgentBenchmark()
    >>> results = benchmark.run_benchmark(
    ...     agent_configs=[
    ...         {"tool_bundle": "minimal"},
    ...         {"tool_bundle": "standard"},
    ...         {"tool_bundle": "full"},
    ...     ],
    ...     scenarios=["simple_peak_count", "decomposition_choice"],
    ... )
    >>>
    >>> # Analyze results
    >>> summary = benchmark.summarize_results(results)
"""

from .runner import AgentBenchmark, BenchmarkResult
from .scenarios import BENCHMARK_SCENARIOS, BenchmarkScenario
from .metrics import compute_agent_metrics, evaluate_response

__all__ = [
    "AgentBenchmark",
    "BenchmarkResult",
    "BENCHMARK_SCENARIOS",
    "BenchmarkScenario",
    "compute_agent_metrics",
    "evaluate_response",
]
