"""Benchmark scenarios for testing agent performance.

This module defines test scenarios that can be used to evaluate
how agents perform with different tool configurations.

Each scenario includes:
- query: The user's question
- expected: What the response should contain
- difficulty: Estimated difficulty
- required_tools: Tools that should be used (for scoring)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class Difficulty(Enum):
    """Difficulty level of a benchmark scenario."""
    SIMPLE = "simple"          # Single tool, direct answer
    MODERATE = "moderate"      # May need 2-3 tools
    COMPLEX = "complex"        # Multi-step reasoning, multiple tools
    CHALLENGING = "challenging"  # Requires tool selection judgment


@dataclass
class ExpectedOutcome:
    """Expected outcome for a benchmark scenario."""

    # Tool usage expectations
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)

    # Response content expectations
    must_contain: List[str] = field(default_factory=list)
    should_contain: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    reasoning_must_contain: List[str] = field(default_factory=list)
    reasoning_should_contain: List[str] = field(default_factory=list)
    reasoning_must_not_contain: List[str] = field(default_factory=list)

    # Format expectations
    expects_number: bool = False
    expects_list: bool = False
    expects_table: bool = False
    expects_recommendation: bool = False

    # Success criteria
    min_tool_calls: int = 0
    max_tool_calls: int = 10


@dataclass
class BenchmarkScenario:
    """A single benchmark scenario for agent testing."""

    name: str
    description: str
    query: str
    expected: ExpectedOutcome
    difficulty: Difficulty = Difficulty.SIMPLE
    category: str = "general"

    # Optional context for the scenario
    context: Optional[str] = None

    # Time limits
    timeout_seconds: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "query": self.query,
            "difficulty": self.difficulty.value,
            "category": self.category,
            "expected": {
                "required_tools": self.expected.required_tools,
                "optional_tools": self.expected.optional_tools,
                "must_contain": self.expected.must_contain,
                "reasoning_must_contain": self.expected.reasoning_must_contain,
                "reasoning_should_contain": self.expected.reasoning_should_contain,
                "expects_number": self.expected.expects_number,
            },
        }


# =============================================================================
# Benchmark Scenario Definitions
# =============================================================================

BENCHMARK_SCENARIOS: Dict[str, BenchmarkScenario] = {}


def register_scenario(scenario: BenchmarkScenario) -> None:
    """Register a benchmark scenario."""
    BENCHMARK_SCENARIOS[scenario.name] = scenario


# -----------------------------------------------------------------------------
# Simple Scenarios (single tool, direct answer)
# -----------------------------------------------------------------------------

register_scenario(BenchmarkScenario(
    name="simple_peak_count",
    description="Count peaks in a time series",
    query="How many peaks are in bx001_real for Re200Rm200?",
    expected=ExpectedOutcome(
        required_tools=["count_peaks", "detect_peaks"],  # Either is acceptable
        must_contain=["peak"],
        expects_number=True,
        min_tool_calls=1,
        max_tool_calls=2,
    ),
    difficulty=Difficulty.SIMPLE,
    category="patterns",
))

register_scenario(BenchmarkScenario(
    name="simple_describe",
    description="Get basic statistics of a series",
    query="What are the basic statistics of by001_real for Re200Rm200?",
    expected=ExpectedOutcome(
        required_tools=["describe_series"],
        must_contain=["mean", "std"],
        should_contain=["min", "max", "rms"],
        min_tool_calls=1,
        max_tool_calls=2,
    ),
    difficulty=Difficulty.SIMPLE,
    category="statistics",
))

register_scenario(BenchmarkScenario(
    name="simple_periodicity",
    description="Detect dominant period in a series",
    query="What is the dominant period in bx001_real for Re200Rm200?",
    expected=ExpectedOutcome(
        required_tools=["detect_periodicity"],
        must_contain=["period"],
        expects_number=True,
        min_tool_calls=1,
        max_tool_calls=2,
    ),
    difficulty=Difficulty.SIMPLE,
    category="spectral",
))

# -----------------------------------------------------------------------------
# Moderate Scenarios (2-3 tools, some reasoning)
# -----------------------------------------------------------------------------

register_scenario(BenchmarkScenario(
    name="decomposition_choice",
    description="Decompose a series and explain the components",
    query="Decompose by001_real for Re200Rm200 and tell me about the trend",
    expected=ExpectedOutcome(
        required_tools=["stl_decompose", "mstl_decompose", "holt_winters_decompose"],
        must_contain=["trend"],
        should_contain=["seasonal", "residual"],
        min_tool_calls=1,
        max_tool_calls=3,
    ),
    difficulty=Difficulty.MODERATE,
    category="decomposition",
))

register_scenario(BenchmarkScenario(
    name="forecast_request",
    description="Generate a forecast",
    query="Forecast the next 50 time steps of bx001_real for Re200Rm200",
    expected=ExpectedOutcome(
        required_tools=["forecast_arima", "forecast_ets", "forecast_theta", "forecast_ensemble"],
        must_contain=["forecast"],
        expects_number=True,
        min_tool_calls=1,
        max_tool_calls=4,
    ),
    difficulty=Difficulty.MODERATE,
    category="forecasting",
))

register_scenario(BenchmarkScenario(
    name="find_patterns",
    description="Find repeating patterns in the data",
    query="Find any repeating patterns or motifs in bx001_real for Re200Rm200",
    expected=ExpectedOutcome(
        required_tools=["analyze_matrix_profile", "find_motifs"],
        must_contain=["motif", "pattern"],
        min_tool_calls=1,
        max_tool_calls=3,
    ),
    difficulty=Difficulty.MODERATE,
    category="patterns",
))

register_scenario(BenchmarkScenario(
    name="detect_anomalies",
    description="Find unusual patterns in the data",
    query="Are there any anomalies or unusual segments in by001_real for Re150Rm150?",
    expected=ExpectedOutcome(
        required_tools=["find_discords", "analyze_matrix_profile"],
        optional_tools=["segment_changepoint", "segment_fluss"],
        must_contain=["discord", "anomal", "unusual"],
        min_tool_calls=1,
        max_tool_calls=4,
    ),
    difficulty=Difficulty.MODERATE,
    category="patterns",
))

# -----------------------------------------------------------------------------
# Complex Scenarios (multi-step, multiple tools)
# -----------------------------------------------------------------------------

register_scenario(BenchmarkScenario(
    name="multi_run_comparison",
    description="Compare statistics across multiple runs",
    query="Compare the RMS values and peak counts across all available runs for bx001_real",
    expected=ExpectedOutcome(
        required_tools=["describe_series", "compare_series_stats"],
        optional_tools=["count_peaks", "detect_peaks"],
        must_contain=["rms", "Re200", "Re175"],
        expects_table=True,
        min_tool_calls=2,
        max_tool_calls=10,
    ),
    difficulty=Difficulty.COMPLEX,
    category="statistics",
))

register_scenario(BenchmarkScenario(
    name="spectral_analysis",
    description="Perform spectral analysis including coherence",
    query="Compute the power spectrum and spectral slope for bx001_real in Re200Rm200. Is this consistent with turbulence?",
    expected=ExpectedOutcome(
        required_tools=["compute_psd"],
        must_contain=["spectrum", "slope"],
        should_contain=["turbulence", "frequency"],
        expects_number=True,
        min_tool_calls=1,
        max_tool_calls=3,
    ),
    difficulty=Difficulty.COMPLEX,
    category="spectral",
))

register_scenario(BenchmarkScenario(
    name="forecast_comparison",
    description="Compare multiple forecasting methods",
    query="What is the best forecasting method for bx001_real in Re200Rm200? Compare at least 2 methods.",
    expected=ExpectedOutcome(
        required_tools=["compare_forecasts"],
        optional_tools=["forecast_arima", "forecast_ets", "forecast_theta", "forecast_ensemble"],
        must_contain=["arima", "ets"],
        reasoning_should_contain=["compare", "confidence", "availability"],
        expects_recommendation=True,
        min_tool_calls=1,
        max_tool_calls=5,
    ),
    difficulty=Difficulty.COMPLEX,
    category="forecasting",
))

# -----------------------------------------------------------------------------
# Challenging Scenarios (requires judgment)
# -----------------------------------------------------------------------------

register_scenario(BenchmarkScenario(
    name="open_analysis",
    description="Open-ended analysis requiring tool selection judgment",
    query="Analyze the dynamics of bx001_real in Re200Rm200. What can you tell me about its behavior?",
    expected=ExpectedOutcome(
        optional_tools=[
            "describe_series", "detect_periodicity", "detect_peaks",
            "stl_decompose", "compute_psd",
        ],
        must_contain=["period", "trend"],
        min_tool_calls=2,
        max_tool_calls=8,
    ),
    difficulty=Difficulty.CHALLENGING,
    category="analysis",
))

register_scenario(BenchmarkScenario(
    name="method_selection",
    description="Choose appropriate method based on data characteristics",
    query="What's the best decomposition method for by001_real in Re200Rm200? Explain why.",
    expected=ExpectedOutcome(
        required_tools=["stl_decompose", "mstl_decompose", "holt_winters_decompose"],
        must_contain=["stl", "trend"],
        reasoning_should_contain=["because", "seasonal", "inspect"],
        expects_recommendation=True,
        min_tool_calls=1,
        max_tool_calls=5,
    ),
    difficulty=Difficulty.CHALLENGING,
    category="decomposition",
))

register_scenario(BenchmarkScenario(
    name="regime_detection",
    description="Detect regime changes in the data",
    query="Are there different regimes or state changes in bx001_real for Re200Rm200? If so, where do they occur?",
    expected=ExpectedOutcome(
        required_tools=["segment_changepoint", "segment_fluss"],
        optional_tools=["analyze_recurrence"],
        must_contain=["segment", "change"],
        min_tool_calls=1,
        max_tool_calls=4,
    ),
    difficulty=Difficulty.CHALLENGING,
    category="patterns",
))


# =============================================================================
# Scenario Groups
# =============================================================================

def get_scenarios_by_difficulty(difficulty: Difficulty) -> List[BenchmarkScenario]:
    """Get all scenarios of a given difficulty level."""
    return [s for s in BENCHMARK_SCENARIOS.values() if s.difficulty == difficulty]


def get_scenarios_by_category(category: str) -> List[BenchmarkScenario]:
    """Get all scenarios in a given category."""
    return [s for s in BENCHMARK_SCENARIOS.values() if s.category == category]


def get_quick_benchmark_scenarios() -> List[str]:
    """Get a quick subset of scenarios for fast testing."""
    return [
        "simple_peak_count",
        "simple_describe",
        "decomposition_choice",
        "forecast_request",
    ]


def get_full_benchmark_scenarios() -> List[str]:
    """Get all scenario names for comprehensive testing."""
    return list(BENCHMARK_SCENARIOS.keys())


def get_scenario_categories() -> List[str]:
    """Get all unique scenario categories."""
    return list(set(s.category for s in BENCHMARK_SCENARIOS.values()))
