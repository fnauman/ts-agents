"""Tests for the experiment log module."""

import tempfile
import shutil
from pathlib import Path

import pytest

from ts_agents.persistence.experiment_log import (
    ExperimentStatus,
    ToolCall,
    ExperimentRun,
    BenchmarkSuite,
    ExperimentLog,
    init_experiment_log,
    get_default_log,
    set_default_log,
)


class TestToolCall:
    """Tests for ToolCall class."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        call = ToolCall(
            tool_name="detect_peaks",
            input_params={"prominence": 0.5},
            duration_ms=150.0,
        )

        assert call.tool_name == "detect_peaks"
        assert call.input_params["prominence"] == 0.5
        assert call.duration_ms == 150.0
        assert call.timestamp is not None

    def test_tool_call_to_dict(self):
        """Test converting tool call to dict."""
        call = ToolCall(tool_name="test", duration_ms=100)
        data = call.to_dict()

        assert data["tool_name"] == "test"
        assert data["duration_ms"] == 100

    def test_tool_call_from_dict(self):
        """Test creating tool call from dict."""
        data = {
            "tool_name": "test",
            "timestamp": "2024-01-01T00:00:00",
            "input_params": {},
            "duration_ms": 100,
        }

        call = ToolCall.from_dict(data)
        assert call.tool_name == "test"


class TestExperimentRun:
    """Tests for ExperimentRun class."""

    def test_run_creation(self):
        """Test creating an experiment run."""
        run = ExperimentRun(
            agent_type="simple",
            model="gpt-4o-mini",
            query="Count the peaks",
        )

        assert run.run_id is not None
        assert len(run.run_id) == 12
        assert run.agent_type == "simple"
        assert run.status == "running"

    def test_add_tool_call(self):
        """Test adding tool calls to a run."""
        run = ExperimentRun(query="Test query")

        run.add_tool_call(ToolCall(tool_name="tool1", duration_ms=100))
        run.add_tool_call(ToolCall(tool_name="tool2", duration_ms=200))

        assert run.tool_call_count == 2
        assert len(run.tool_calls) == 2

    def test_add_message(self):
        """Test adding messages to a run."""
        run = ExperimentRun(query="Test query")

        run.add_message("user", "Hello")
        run.add_message("assistant", "Hi there")

        assert len(run.messages) == 2
        assert run.messages[0]["role"] == "user"

    def test_complete_run(self):
        """Test completing a run."""
        run = ExperimentRun(query="Test query")

        run.complete(
            response="Found 42 peaks",
            tokens={"input": 100, "output": 50, "total": 150},
        )

        assert run.status == ExperimentStatus.COMPLETED.value
        assert run.response == "Found 42 peaks"
        assert run.tokens_total == 150
        assert run.ended is not None
        assert run.total_duration_ms > 0

    def test_fail_run(self):
        """Test failing a run."""
        run = ExperimentRun(query="Test query")

        run.fail("Something went wrong")

        assert run.status == ExperimentStatus.FAILED.value
        assert run.error == "Something went wrong"
        assert run.ended is not None

    def test_run_to_dict(self):
        """Test converting run to dict."""
        run = ExperimentRun(agent_type="simple", query="Test")
        data = run.to_dict()

        assert data["agent_type"] == "simple"
        assert data["query"] == "Test"


class TestExperimentLog:
    """Tests for ExperimentLog class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def log(self, temp_log_dir):
        """Create a log instance for testing."""
        return ExperimentLog(root_dir=temp_log_dir)

    def test_log_initialization(self, temp_log_dir):
        """Test log initializes correctly."""
        log = ExperimentLog(root_dir=temp_log_dir)

        assert log.root.exists()
        assert log.runs_dir.exists()
        assert log.benchmarks_dir.exists()

    def test_save_and_load_run(self, log):
        """Test saving and loading a run."""
        run = ExperimentRun(
            agent_type="simple",
            model="gpt-4o-mini",
            query="Count the peaks",
        )
        run.add_tool_call(ToolCall(tool_name="detect_peaks"))
        run.complete("Found 42 peaks")

        run_id = log.save_run(run)
        loaded = log.load_run(run_id)

        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert loaded.agent_type == "simple"
        assert loaded.response == "Found 42 peaks"
        assert loaded.tool_call_count == 1

    def test_load_nonexistent(self, log):
        """Test loading a non-existent run."""
        loaded = log.load_run("nonexistent")
        assert loaded is None

    def test_list_runs(self, log):
        """Test listing runs."""
        run1 = ExperimentRun(agent_type="simple", model="model1")
        run1.complete("result1")
        log.save_run(run1)

        run2 = ExperimentRun(agent_type="deep", model="model2")
        run2.complete("result2")
        log.save_run(run2)

        # List all
        all_runs = log.list_runs()
        assert len(all_runs) == 2

        # Filter by agent type
        simple_runs = log.list_runs(agent_type="simple")
        assert len(simple_runs) == 1
        assert simple_runs[0]["agent_type"] == "simple"

        # Filter by model
        model1_runs = log.list_runs(model="model1")
        assert len(model1_runs) == 1

    def test_list_runs_with_tags(self, log):
        """Test listing runs filtered by tags."""
        run1 = ExperimentRun(query="Test 1", tags=["benchmark", "fast"])
        log.save_run(run1)

        run2 = ExperimentRun(query="Test 2", tags=["benchmark", "slow"])
        log.save_run(run2)

        run3 = ExperimentRun(query="Test 3", tags=["production"])
        log.save_run(run3)

        benchmark_runs = log.list_runs(tags=["benchmark"])
        assert len(benchmark_runs) == 2

    def test_delete_run(self, log):
        """Test deleting a run."""
        run = ExperimentRun(query="Test")
        log.save_run(run)

        result = log.delete_run(run.run_id)
        assert result is True

        loaded = log.load_run(run.run_id)
        assert loaded is None

    def test_max_runs_cleanup(self, temp_log_dir):
        """Test that old runs are cleaned up when over limit."""
        log = ExperimentLog(root_dir=temp_log_dir, max_runs=3)

        for i in range(5):
            run = ExperimentRun(query=f"Query {i}")
            log.save_run(run)

        runs = log.list_runs()
        assert len(runs) <= 3

    def test_get_stats(self, log):
        """Test getting run statistics."""
        run1 = ExperimentRun(agent_type="simple")
        run1.complete("result", tokens={"input": 100, "output": 50, "total": 150})
        run1.add_tool_call(ToolCall(tool_name="tool1"))
        log.save_run(run1)

        run2 = ExperimentRun(agent_type="simple")
        run2.fail("error")
        log.save_run(run2)

        stats = log.get_stats(agent_type="simple")

        assert stats["total_runs"] == 2
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["total_tokens"] == 150


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def log(self, temp_log_dir):
        """Create a log instance for testing."""
        return ExperimentLog(root_dir=temp_log_dir)

    def test_benchmark_suite_creation(self):
        """Test creating a benchmark suite."""
        suite = BenchmarkSuite(
            name="Tool Scaling Test",
            description="Testing agent with varying tool counts",
        )

        assert suite.suite_id is not None
        assert suite.name == "Tool Scaling Test"

    def test_save_and_load_benchmark(self, log):
        """Test saving and loading a benchmark suite."""
        suite = BenchmarkSuite(
            name="Test Suite",
            configurations=[{"tools": 5}, {"tools": 10}],
            results=[{"accuracy": 0.9}, {"accuracy": 0.85}],
        )

        log.save_benchmark(suite)
        loaded = log.load_benchmark(suite.suite_id)

        assert loaded is not None
        assert loaded.name == "Test Suite"
        assert len(loaded.results) == 2

    def test_list_benchmarks(self, log):
        """Test listing benchmark suites."""
        suite1 = BenchmarkSuite(name="Suite 1")
        log.save_benchmark(suite1)

        suite2 = BenchmarkSuite(name="Suite 2")
        log.save_benchmark(suite2)

        benchmarks = log.list_benchmarks()
        assert len(benchmarks) == 2


class TestDefaultLog:
    """Tests for default log management."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        set_default_log(None)

    def test_init_experiment_log(self, temp_log_dir):
        """Test init_experiment_log sets the default log."""
        log = init_experiment_log(root_dir=temp_log_dir)

        default = get_default_log()
        assert default is log
