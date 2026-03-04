"""Tests for benchmark module."""

import pytest


class TestBenchmarkScenarios:
    """Tests for benchmark scenario definitions."""

    def test_scenarios_registered(self):
        """Test that scenarios are registered."""
        from src.agents.benchmarks.scenarios import BENCHMARK_SCENARIOS

        assert len(BENCHMARK_SCENARIOS) > 0

    def test_simple_peak_count_scenario(self):
        """Test simple_peak_count scenario definition."""
        from src.agents.benchmarks.scenarios import BENCHMARK_SCENARIOS

        scenario = BENCHMARK_SCENARIOS.get("simple_peak_count")

        assert scenario is not None
        assert "peak" in scenario.query.lower()
        assert len(scenario.expected.required_tools) > 0

    def test_decomposition_choice_scenario(self):
        """Test decomposition_choice scenario definition."""
        from src.agents.benchmarks.scenarios import BENCHMARK_SCENARIOS

        scenario = BENCHMARK_SCENARIOS.get("decomposition_choice")

        assert scenario is not None
        assert "decompos" in scenario.query.lower()

    def test_scenario_has_expected_outcome(self):
        """Test that all scenarios have expected outcomes."""
        from src.agents.benchmarks.scenarios import BENCHMARK_SCENARIOS

        for name, scenario in BENCHMARK_SCENARIOS.items():
            assert scenario.expected is not None
            assert hasattr(scenario.expected, "required_tools")
            assert hasattr(scenario.expected, "must_contain")

    def test_scenario_to_dict(self):
        """Test scenario serialization."""
        from src.agents.benchmarks.scenarios import BENCHMARK_SCENARIOS

        scenario = BENCHMARK_SCENARIOS["simple_peak_count"]
        data = scenario.to_dict()

        assert "name" in data
        assert "query" in data
        assert "difficulty" in data
        assert "expected" in data

    def test_get_scenarios_by_difficulty(self):
        """Test filtering scenarios by difficulty."""
        from src.agents.benchmarks.scenarios import (
            get_scenarios_by_difficulty,
            Difficulty,
        )

        simple = get_scenarios_by_difficulty(Difficulty.SIMPLE)
        complex_ = get_scenarios_by_difficulty(Difficulty.COMPLEX)

        assert len(simple) > 0
        for s in simple:
            assert s.difficulty == Difficulty.SIMPLE

        assert len(complex_) > 0
        for s in complex_:
            assert s.difficulty == Difficulty.COMPLEX

    def test_get_scenarios_by_category(self):
        """Test filtering scenarios by category."""
        from src.agents.benchmarks.scenarios import get_scenarios_by_category

        patterns = get_scenarios_by_category("patterns")

        assert len(patterns) > 0
        for s in patterns:
            assert s.category == "patterns"

    def test_get_quick_benchmark_scenarios(self):
        """Test getting quick benchmark scenario list."""
        from src.agents.benchmarks.scenarios import (
            get_quick_benchmark_scenarios,
            BENCHMARK_SCENARIOS,
        )

        quick = get_quick_benchmark_scenarios()

        assert len(quick) > 0
        assert len(quick) < len(BENCHMARK_SCENARIOS)
        for name in quick:
            assert name in BENCHMARK_SCENARIOS

    def test_get_full_benchmark_scenarios(self):
        """Test getting full benchmark scenario list."""
        from src.agents.benchmarks.scenarios import (
            get_full_benchmark_scenarios,
            BENCHMARK_SCENARIOS,
        )

        full = get_full_benchmark_scenarios()

        assert len(full) == len(BENCHMARK_SCENARIOS)

    def test_get_scenario_categories(self):
        """Test getting all categories."""
        from src.agents.benchmarks.scenarios import get_scenario_categories

        categories = get_scenario_categories()

        assert len(categories) > 0
        assert "patterns" in categories
        assert "forecasting" in categories


class TestExpectedOutcome:
    """Tests for ExpectedOutcome dataclass."""

    def test_expected_outcome_defaults(self):
        """Test ExpectedOutcome default values."""
        from src.agents.benchmarks.scenarios import ExpectedOutcome

        expected = ExpectedOutcome()

        assert expected.required_tools == []
        assert expected.optional_tools == []
        assert expected.must_contain == []
        assert expected.expects_number is False
        assert expected.min_tool_calls == 0
        assert expected.max_tool_calls == 10

    def test_expected_outcome_with_values(self):
        """Test ExpectedOutcome with custom values."""
        from src.agents.benchmarks.scenarios import ExpectedOutcome

        expected = ExpectedOutcome(
            required_tools=["detect_peaks"],
            must_contain=["peak"],
            expects_number=True,
            min_tool_calls=1,
        )

        assert expected.required_tools == ["detect_peaks"]
        assert expected.must_contain == ["peak"]
        assert expected.expects_number is True
        assert expected.min_tool_calls == 1


class TestMetrics:
    """Tests for benchmark metrics."""

    def test_evaluate_response_basic(self):
        """Test basic response evaluation."""
        from src.agents.benchmarks.metrics import evaluate_response
        from src.agents.benchmarks.scenarios import ExpectedOutcome

        expected = ExpectedOutcome(
            required_tools=["detect_peaks"],
            must_contain=["peak"],
            expects_number=True,
        )

        result = evaluate_response(
            response="Found 10 peaks in the data.",
            tool_calls=["detect_peaks"],
            expected=expected,
        )

        assert result.passed is True
        assert result.tool_score > 0
        assert result.content_score > 0
        assert "detect_peaks" in result.required_tools_used

    def test_evaluate_response_missing_tool(self):
        """Test evaluation when required tool is missing."""
        from src.agents.benchmarks.metrics import evaluate_response
        from src.agents.benchmarks.scenarios import ExpectedOutcome

        expected = ExpectedOutcome(
            required_tools=["detect_peaks"],
            must_contain=["peak"],
        )

        result = evaluate_response(
            response="The data shows peaks.",
            tool_calls=["stl_decompose"],  # Wrong tool
            expected=expected,
        )

        assert "detect_peaks" in result.required_tools_missed
        assert result.tool_score < 1.0

    def test_evaluate_response_missing_content(self):
        """Test evaluation when required content is missing."""
        from src.agents.benchmarks.metrics import evaluate_response
        from src.agents.benchmarks.scenarios import ExpectedOutcome

        expected = ExpectedOutcome(
            required_tools=["detect_peaks"],
            must_contain=["peak", "count"],
        )

        result = evaluate_response(
            response="Analysis complete.",  # Missing required content
            tool_calls=["detect_peaks"],
            expected=expected,
        )

        assert len(result.content_misses) > 0
        assert result.content_score < 1.0

    def test_evaluate_response_forbidden_tool(self):
        """Test evaluation when forbidden tool is used."""
        from src.agents.benchmarks.metrics import evaluate_response
        from src.agents.benchmarks.scenarios import ExpectedOutcome

        expected = ExpectedOutcome(
            required_tools=["detect_peaks"],
            forbidden_tools=["forecast_arima"],
            must_contain=["peak"],
        )

        result = evaluate_response(
            response="Found 5 peaks.",
            tool_calls=["detect_peaks", "forecast_arima"],  # Forbidden tool used
            expected=expected,
        )

        assert "forecast_arima" in result.forbidden_tools_used
        assert result.passed is False

    def test_evaluate_response_expects_number(self):
        """Test evaluation of number expectation."""
        from src.agents.benchmarks.metrics import evaluate_response
        from src.agents.benchmarks.scenarios import ExpectedOutcome

        expected = ExpectedOutcome(
            must_contain=[],
            expects_number=True,
        )

        result_with_number = evaluate_response(
            response="There are 42 items.",
            tool_calls=[],
            expected=expected,
        )

        result_without_number = evaluate_response(
            response="There are many items.",
            tool_calls=[],
            expected=expected,
        )

        assert result_with_number.format_score > result_without_number.format_score

    def test_evaluation_result_dataclass(self):
        """Test EvaluationResult dataclass."""
        from src.agents.benchmarks.metrics import EvaluationResult

        result = EvaluationResult(
            tool_score=0.8,
            content_score=0.9,
            format_score=1.0,
            overall_score=0.87,
            tools_used=["detect_peaks"],
            passed=True,
        )

        assert result.tool_score == 0.8
        assert result.passed is True
        assert len(result.failure_reasons) == 0

    def test_summarize_evaluations(self):
        """Test summarizing multiple evaluations."""
        from src.agents.benchmarks.metrics import (
            EvaluationResult,
            summarize_evaluations,
        )

        evaluations = [
            EvaluationResult(
                tool_score=0.8, content_score=0.9,
                format_score=1.0, overall_score=0.87, passed=True,
            ),
            EvaluationResult(
                tool_score=0.5, content_score=0.5,
                format_score=0.5, overall_score=0.5, passed=False,
                failure_reasons=["Missing content"],
            ),
        ]

        summary = summarize_evaluations(evaluations)

        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5
        assert "avg_overall_score" in summary

    def test_summarize_evaluations_empty(self):
        """Test summarizing empty evaluation list."""
        from src.agents.benchmarks.metrics import summarize_evaluations

        summary = summarize_evaluations([])

        assert "error" in summary


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a benchmark result."""
        from src.agents.benchmarks.runner import BenchmarkResult

        result = BenchmarkResult(
            scenario_name="simple_peak_count",
            model_name="gpt-4o-mini",
            tool_bundle="standard",
            tool_count=15,
            success=True,
            response="Found 10 peaks.",
            tool_calls=["detect_peaks"],
            duration_ms=1500.0,
        )

        assert result.scenario_name == "simple_peak_count"
        assert result.success is True
        assert result.tool_count == 15

    def test_benchmark_result_to_dict(self):
        """Test serializing benchmark result."""
        from src.agents.benchmarks.runner import BenchmarkResult
        from src.agents.benchmarks.metrics import EvaluationResult

        result = BenchmarkResult(
            scenario_name="simple_peak_count",
            model_name="gpt-4o-mini",
            tool_bundle="standard",
            tool_count=15,
            success=True,
            response="Found 10 peaks.",
            tool_calls=["detect_peaks"],
            duration_ms=1500.0,
            evaluation=EvaluationResult(
                tool_score=1.0, content_score=1.0,
                format_score=1.0, overall_score=1.0, passed=True,
            ),
        )

        data = result.to_dict()

        assert data["scenario_name"] == "simple_peak_count"
        assert data["success"] is True
        assert data["evaluation"]["passed"] is True


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_agent_config_defaults(self):
        """Test AgentConfig default values."""
        from src.agents.benchmarks.runner import AgentConfig

        config = AgentConfig()

        assert config.tool_bundle == "standard"
        assert config.model_name is None
        assert config.temperature == 0

    def test_agent_config_get_name(self):
        """Test AgentConfig name generation."""
        from src.agents.benchmarks.runner import AgentConfig

        config1 = AgentConfig(tool_bundle="minimal")
        assert config1.get_name() == "minimal"

        config2 = AgentConfig(tool_bundle="standard", model_name="gpt-4o")
        assert "standard" in config2.get_name()
        assert "gpt-4o" in config2.get_name()

        config3 = AgentConfig(tool_bundle="full", name="Custom Name")
        assert config3.get_name() == "Custom Name"


class TestAgentBenchmarkInfrastructure:
    """Tests for AgentBenchmark infrastructure (without actual LLM calls)."""

    def test_benchmark_class_exists(self):
        """Test that AgentBenchmark class can be instantiated."""
        from src.agents.benchmarks.runner import AgentBenchmark
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = AgentBenchmark(output_dir=tmpdir)

            assert benchmark.output_dir.exists()
            assert benchmark.verbose is True

    def test_summarize_results_empty(self):
        """Test summarizing empty results."""
        from src.agents.benchmarks.runner import AgentBenchmark
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = AgentBenchmark(output_dir=tmpdir)

            summary = benchmark.summarize_results([])

            assert summary["total_runs"] == 0

    def test_summarize_results_with_data(self):
        """Test summarizing results with sample data."""
        from src.agents.benchmarks.runner import AgentBenchmark, BenchmarkResult
        from src.agents.benchmarks.metrics import EvaluationResult
        import tempfile

        results = [
            BenchmarkResult(
                scenario_name="test1",
                model_name="model1",
                tool_bundle="minimal",
                tool_count=5,
                success=True,
                response="Response 1",
                tool_calls=["tool1"],
                duration_ms=100.0,
                evaluation=EvaluationResult(
                    tool_score=1.0, content_score=1.0,
                    format_score=1.0, overall_score=1.0, passed=True,
                ),
            ),
            BenchmarkResult(
                scenario_name="test1",
                model_name="model1",
                tool_bundle="standard",
                tool_count=15,
                success=True,
                response="Response 2",
                tool_calls=["tool1", "tool2"],
                duration_ms=200.0,
                evaluation=EvaluationResult(
                    tool_score=0.5, content_score=0.5,
                    format_score=0.5, overall_score=0.5, passed=False,
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = AgentBenchmark(output_dir=tmpdir)

            summary = benchmark.summarize_results(results)

            assert summary["total_runs"] == 2
            assert summary["total_pass"] == 1
            assert "by_config" in summary
            assert "by_scenario" in summary

    def test_export_results(self):
        """Test exporting results to file."""
        from src.agents.benchmarks.runner import AgentBenchmark, BenchmarkResult
        import tempfile
        import json

        results = [
            BenchmarkResult(
                scenario_name="test1",
                model_name="model1",
                tool_bundle="minimal",
                tool_count=5,
                success=True,
                response="Test",
                tool_calls=[],
                duration_ms=100.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = AgentBenchmark(output_dir=tmpdir)

            filepath = benchmark.export_results(results, "test_export.json")

            assert "test_export.json" in filepath

            with open(filepath) as f:
                data = json.load(f)

            assert "results" in data
            assert len(data["results"]) == 1


class TestConvenienceFunctions:
    """Tests for benchmark convenience functions."""

    def test_run_quick_benchmark_function_exists(self):
        """Test that run_quick_benchmark function exists."""
        from src.agents.benchmarks.runner import run_quick_benchmark

        assert callable(run_quick_benchmark)

    def test_run_full_benchmark_function_exists(self):
        """Test that run_full_benchmark function exists."""
        from src.agents.benchmarks.runner import run_full_benchmark

        assert callable(run_full_benchmark)
