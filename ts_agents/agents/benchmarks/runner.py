"""Benchmark runner for agent performance analysis.

This module provides the main infrastructure for running benchmark
scenarios across different agent configurations.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..simple import create_simple_agent, SimpleAgentChat
from .scenarios import (
    BENCHMARK_SCENARIOS,
    BenchmarkScenario,
    get_quick_benchmark_scenarios,
    get_full_benchmark_scenarios,
)
from .metrics import evaluate_response, compute_agent_metrics, EvaluationResult


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    # Configuration
    scenario_name: str
    model_name: str
    tool_bundle: str
    tool_count: int

    # Outcome
    success: bool
    response: str
    tool_calls: List[str]

    # Performance
    duration_ms: float
    token_count: Optional[int] = None

    # Evaluation
    evaluation: Optional[EvaluationResult] = None
    error: Optional[str] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_name": self.scenario_name,
            "model_name": self.model_name,
            "tool_bundle": self.tool_bundle,
            "tool_count": self.tool_count,
            "success": self.success,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "duration_ms": self.duration_ms,
            "token_count": self.token_count,
            "evaluation": {
                "tool_score": self.evaluation.tool_score,
                "content_score": self.evaluation.content_score,
                "reasoning_score": self.evaluation.reasoning_score,
                "format_score": self.evaluation.format_score,
                "overall_score": self.evaluation.overall_score,
                "passed": self.evaluation.passed,
                "failure_reasons": self.evaluation.failure_reasons,
            } if self.evaluation else None,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentConfig:
    """Configuration for an agent to benchmark."""
    tool_bundle: str = "standard"
    model_name: Optional[str] = None
    custom_tools: Optional[List[str]] = None
    temperature: float = 0
    name: Optional[str] = None  # Display name

    def get_name(self) -> str:
        """Get display name for this config."""
        if self.name:
            return self.name
        parts = [self.tool_bundle]
        if self.model_name:
            parts.append(self.model_name.split("/")[-1])
        return "-".join(parts)


class AgentBenchmark:
    """Benchmark runner for comparing agent configurations.

    This class provides infrastructure to:
    - Run multiple scenarios across multiple agent configurations
    - Collect detailed metrics and evaluations
    - Export results for analysis
    - Generate summary reports

    Parameters
    ----------
    output_dir : str
        Directory to save benchmark results
    verbose : bool
        Print progress during benchmark runs

    Examples
    --------
    >>> benchmark = AgentBenchmark(output_dir="./experiments/benchmarks")
    >>>
    >>> # Quick benchmark comparing tool bundles
    >>> results = benchmark.run_bundle_comparison(
    ...     bundles=["minimal", "standard", "full"],
    ...     scenarios=["simple_peak_count", "decomposition_choice"],
    ... )
    >>>
    >>> # Full benchmark with multiple configs
    >>> results = benchmark.run_benchmark(
    ...     configs=[
    ...         AgentConfig(tool_bundle="minimal"),
    ...         AgentConfig(tool_bundle="standard"),
    ...         AgentConfig(tool_bundle="full"),
    ...     ],
    ...     scenarios=get_full_benchmark_scenarios(),
    ... )
    >>>
    >>> # Analyze results
    >>> summary = benchmark.summarize_results(results)
    >>> benchmark.export_results(results, "benchmark_results.json")
    """

    def __init__(
        self,
        output_dir: str = "./experiments/benchmarks",
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def run_benchmark(
        self,
        configs: List[AgentConfig],
        scenarios: Optional[List[str]] = None,
        n_runs: int = 1,
        timeout_seconds: int = 120,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmark scenarios across multiple agent configurations.

        Parameters
        ----------
        configs : List[AgentConfig]
            Agent configurations to test
        scenarios : List[str], optional
            Scenario names to run (default: quick benchmark scenarios)
        n_runs : int
            Number of times to run each scenario per config
        timeout_seconds : int
            Maximum time per scenario
        progress_callback : Callable, optional
            Called with (current, total, description) for progress updates

        Returns
        -------
        List[BenchmarkResult]
            Results for all runs
        """
        if scenarios is None:
            scenarios = get_quick_benchmark_scenarios()

        total_runs = len(configs) * len(scenarios) * n_runs
        current_run = 0
        results = []

        for config in configs:
            if self.verbose:
                logger.info(f"Testing config: {config.get_name()}")

            for scenario_name in scenarios:
                scenario = BENCHMARK_SCENARIOS.get(scenario_name)
                if scenario is None:
                    logger.warning(f"Unknown scenario: {scenario_name}")
                    continue

                for run_idx in range(n_runs):
                    current_run += 1

                    if progress_callback:
                        progress_callback(
                            current_run, total_runs,
                            f"{config.get_name()} / {scenario_name}"
                        )

                    if self.verbose:
                        print(f"  [{current_run}/{total_runs}] {scenario_name}...", end=" ")

                    result = self._run_single_benchmark(
                        config=config,
                        scenario=scenario,
                        timeout_seconds=timeout_seconds,
                    )
                    results.append(result)

                    if self.verbose:
                        status = "PASS" if result.evaluation and result.evaluation.passed else "FAIL"
                        score = result.evaluation.overall_score if result.evaluation else 0
                        print(f"{status} ({score:.2f}) - {result.duration_ms:.0f}ms")

        return results

    def _run_single_benchmark(
        self,
        config: AgentConfig,
        scenario: BenchmarkScenario,
        timeout_seconds: int,
    ) -> BenchmarkResult:
        """Run a single benchmark scenario."""
        tool_calls: List[str] = []

        def log_callback(entry: Dict[str, Any]):
            if entry.get("event") == "tool_call":
                tool_calls.append(entry["tool_name"])

        try:
            # Create agent with logging
            agent = create_simple_agent(
                model_name=config.model_name,
                tool_bundle=config.tool_bundle,
                custom_tools=config.custom_tools,
                temperature=config.temperature,
                enable_logging=True,
                log_callback=log_callback,
            )

            tool_count = len(agent._ts_agents_metadata.get("tool_names", []))

            # Run query
            start_time = time.time()
            from langchain_core.messages import HumanMessage, AIMessage

            result = agent.invoke({
                "messages": [HumanMessage(content=scenario.query)]
            })
            duration_ms = (time.time() - start_time) * 1000

            # Extract response
            messages = result.get("messages", [])
            response = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    response = msg.content
                    break

            # Evaluate response
            evaluation = evaluate_response(
                response=response,
                tool_calls=tool_calls,
                expected=scenario.expected,
            )

            return BenchmarkResult(
                scenario_name=scenario.name,
                model_name=config.model_name or "default",
                tool_bundle=config.tool_bundle,
                tool_count=tool_count,
                success=True,
                response=response,
                tool_calls=tool_calls,
                duration_ms=duration_ms,
                evaluation=evaluation,
            )

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return BenchmarkResult(
                scenario_name=scenario.name,
                model_name=config.model_name or "default",
                tool_bundle=config.tool_bundle,
                tool_count=0,
                success=False,
                response="",
                tool_calls=tool_calls,
                duration_ms=0,
                error=str(e),
            )

    def run_bundle_comparison(
        self,
        bundles: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        n_runs: int = 1,
    ) -> List[BenchmarkResult]:
        """Convenience method to compare different tool bundles.

        Parameters
        ----------
        bundles : List[str], optional
            Bundles to compare (default: minimal, standard, full)
        scenarios : List[str], optional
            Scenarios to run
        model_name : str, optional
            Model to use for all configs
        n_runs : int
            Number of runs per scenario

        Returns
        -------
        List[BenchmarkResult]
            Results for all runs
        """
        if bundles is None:
            bundles = ["minimal", "standard", "full"]

        configs = [
            AgentConfig(tool_bundle=bundle, model_name=model_name)
            for bundle in bundles
        ]

        return self.run_benchmark(
            configs=configs,
            scenarios=scenarios,
            n_runs=n_runs,
        )

    def summarize_results(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results.

        Parameters
        ----------
        results : List[BenchmarkResult]
            Benchmark results to summarize

        Returns
        -------
        Dict[str, Any]
            Summary statistics by config and scenario
        """
        # Group by config
        by_config: Dict[str, List[BenchmarkResult]] = {}
        for r in results:
            key = f"{r.tool_bundle}_{r.model_name}"
            if key not in by_config:
                by_config[key] = []
            by_config[key].append(r)

        # Group by scenario
        by_scenario: Dict[str, List[BenchmarkResult]] = {}
        for r in results:
            if r.scenario_name not in by_scenario:
                by_scenario[r.scenario_name] = []
            by_scenario[r.scenario_name].append(r)

        # Compute summaries
        config_summaries = {}
        for config_key, config_results in by_config.items():
            valid = [r for r in config_results if r.evaluation]
            config_summaries[config_key] = {
                "total": len(config_results),
                "success": sum(1 for r in config_results if r.success),
                "pass_rate": sum(1 for r in valid if r.evaluation.passed) / len(valid) if valid else 0,
                "avg_score": sum(r.evaluation.overall_score for r in valid) / len(valid) if valid else 0,
                "avg_duration_ms": sum(r.duration_ms for r in config_results) / len(config_results),
                "tool_count": config_results[0].tool_count if config_results else 0,
                "avg_tool_calls": sum(len(r.tool_calls) for r in config_results) / len(config_results),
            }

        scenario_summaries = {}
        for scenario_name, scenario_results in by_scenario.items():
            valid = [r for r in scenario_results if r.evaluation]
            scenario_summaries[scenario_name] = {
                "total": len(scenario_results),
                "pass_rate": sum(1 for r in valid if r.evaluation.passed) / len(valid) if valid else 0,
                "avg_score": sum(r.evaluation.overall_score for r in valid) / len(valid) if valid else 0,
                "by_bundle": {
                    bundle: {
                        "pass_rate": sum(1 for r in valid if r.evaluation.passed and r.tool_bundle == bundle)
                                   / sum(1 for r in valid if r.tool_bundle == bundle)
                        if sum(1 for r in valid if r.tool_bundle == bundle) > 0 else 0,
                    }
                    for bundle in set(r.tool_bundle for r in scenario_results)
                }
            }

        return {
            "total_runs": len(results),
            "total_success": sum(1 for r in results if r.success),
            "total_pass": sum(1 for r in results if r.evaluation and r.evaluation.passed),
            "by_config": config_summaries,
            "by_scenario": scenario_summaries,
        }

    def export_results(
        self,
        results: List[BenchmarkResult],
        filename: Optional[str] = None,
    ) -> str:
        """Export benchmark results to JSON file.

        Parameters
        ----------
        results : List[BenchmarkResult]
            Results to export
        filename : str, optional
            Output filename (default: auto-generated with timestamp)

        Returns
        -------
        str
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"

        filepath = self.output_dir / filename

        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_runs": len(results),
            },
            "results": [r.to_dict() for r in results],
            "summary": self.summarize_results(results),
        }

        filepath.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Results exported to {filepath}")

        return str(filepath)

    def print_summary(
        self,
        results: List[BenchmarkResult],
    ) -> None:
        """Print a formatted summary of benchmark results.

        Parameters
        ----------
        results : List[BenchmarkResult]
            Results to summarize
        """
        summary = self.summarize_results(results)

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"\nTotal runs: {summary['total_runs']}")
        print(f"Success rate: {summary['total_success'] / summary['total_runs'] * 100:.1f}%")
        print(f"Pass rate: {summary['total_pass'] / summary['total_runs'] * 100:.1f}%")

        print("\n" + "-" * 60)
        print("BY CONFIGURATION")
        print("-" * 60)

        for config, stats in summary["by_config"].items():
            print(f"\n{config}:")
            print(f"  Tool count: {stats['tool_count']}")
            print(f"  Pass rate: {stats['pass_rate'] * 100:.1f}%")
            print(f"  Avg score: {stats['avg_score']:.3f}")
            print(f"  Avg duration: {stats['avg_duration_ms']:.0f}ms")
            print(f"  Avg tool calls: {stats['avg_tool_calls']:.1f}")

        print("\n" + "-" * 60)
        print("BY SCENARIO")
        print("-" * 60)

        for scenario, stats in summary["by_scenario"].items():
            print(f"\n{scenario}:")
            print(f"  Pass rate: {stats['pass_rate'] * 100:.1f}%")
            print(f"  Avg score: {stats['avg_score']:.3f}")

        print("\n" + "=" * 60)


# =============================================================================
# Convenience Functions
# =============================================================================

def run_quick_benchmark(
    bundles: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a quick benchmark comparing tool bundles.

    Parameters
    ----------
    bundles : List[str], optional
        Bundles to compare
    verbose : bool
        Print progress

    Returns
    -------
    Dict[str, Any]
        Summary of benchmark results
    """
    benchmark = AgentBenchmark(verbose=verbose)
    results = benchmark.run_bundle_comparison(
        bundles=bundles,
        scenarios=get_quick_benchmark_scenarios(),
    )
    benchmark.print_summary(results)
    return benchmark.summarize_results(results)


def run_full_benchmark(
    bundles: Optional[List[str]] = None,
    n_runs: int = 1,
    export: bool = True,
) -> Dict[str, Any]:
    """Run a comprehensive benchmark.

    Parameters
    ----------
    bundles : List[str], optional
        Bundles to compare
    n_runs : int
        Number of runs per scenario
    export : bool
        Whether to export results to file

    Returns
    -------
    Dict[str, Any]
        Summary of benchmark results
    """
    benchmark = AgentBenchmark()
    results = benchmark.run_bundle_comparison(
        bundles=bundles,
        scenarios=get_full_benchmark_scenarios(),
        n_runs=n_runs,
    )

    if export:
        benchmark.export_results(results)

    benchmark.print_summary(results)
    return benchmark.summarize_results(results)
