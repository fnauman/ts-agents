"""Experiment logging for agent runs and benchmarks.

This module provides structured logging of agent experiments,
including tool calls, responses, and performance metrics.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ToolCall:
    """Record of a single tool call during an experiment."""
    tool_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_params: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentRun:
    """Record of a complete agent experiment run.

    Captures all details of an agent interaction for later analysis,
    including the query, all tool calls, final response, and metrics.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    started: str = field(default_factory=lambda: datetime.now().isoformat())
    ended: Optional[str] = None
    status: str = "running"

    # Configuration
    agent_type: str = "simple"  # "simple" or "deep"
    model: str = "unknown"
    tool_bundle: str = "standard"

    # Input
    query: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Execution trace
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)

    # Output
    response: str = ""
    error: Optional[str] = None

    # Metrics
    total_duration_ms: float = 0.0
    tool_call_count: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0

    # Tags for filtering
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def add_tool_call(self, call: ToolCall) -> None:
        """Add a tool call to the trace."""
        self.tool_calls.append(call.to_dict())
        self.tool_call_count = len(self.tool_calls)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the trace."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def complete(self, response: str, tokens: Optional[Dict[str, int]] = None) -> None:
        """Mark the run as completed."""
        self.ended = datetime.now().isoformat()
        self.status = ExperimentStatus.COMPLETED.value
        self.response = response

        if tokens:
            self.tokens_input = tokens.get("input", 0)
            self.tokens_output = tokens.get("output", 0)
            self.tokens_total = tokens.get("total", 0)

        # Calculate total duration
        started = datetime.fromisoformat(self.started)
        ended = datetime.fromisoformat(self.ended)
        self.total_duration_ms = (ended - started).total_seconds() * 1000

    def fail(self, error: str) -> None:
        """Mark the run as failed."""
        self.ended = datetime.now().isoformat()
        self.status = ExperimentStatus.FAILED.value
        self.error = error

        started = datetime.fromisoformat(self.started)
        ended = datetime.fromisoformat(self.ended)
        self.total_duration_ms = (ended - started).total_seconds() * 1000


@dataclass
class BenchmarkSuite:
    """A collection of benchmark results."""
    suite_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration tested
    configurations: List[Dict[str, Any]] = field(default_factory=list)

    # Results
    results: List[Dict[str, Any]] = field(default_factory=list)

    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSuite":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ExperimentLog:
    """Persistent log of agent experiments.

    Stores detailed records of agent runs for analysis and benchmarking.
    Supports filtering, aggregation, and export.

    Parameters
    ----------
    root_dir : str or Path
        Root directory for experiment logs
    max_runs : int
        Maximum runs to retain (oldest deleted first)

    Examples
    --------
    >>> log = ExperimentLog("./experiments")
    >>> run = ExperimentRun(
    ...     agent_type="simple",
    ...     query="Analyze the peaks in bx001_real"
    ... )
    >>> run.add_tool_call(ToolCall(tool_name="detect_peaks", duration_ms=150))
    >>> run.complete("Found 42 peaks...")
    >>> log.save_run(run)
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "./experiments",
        max_runs: int = 1000,
    ):
        self.root = Path(root_dir)
        self.runs_dir = self.root / "runs"
        self.benchmarks_dir = self.root / "benchmarks"
        self.max_runs = max_runs
        self._runs_index: Dict[str, Dict[str, Any]] = {}

        # Create directories
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        self.benchmarks_dir.mkdir(exist_ok=True)

        self._load_index()

    @property
    def index_file(self) -> Path:
        """Path to the runs index file."""
        return self.root / "runs_index.json"

    def _load_index(self) -> None:
        """Load the runs index from disk."""
        if self.index_file.exists():
            try:
                self._runs_index = json.loads(self.index_file.read_text())
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load runs index: {e}")
                self._runs_index = {}
        else:
            self._runs_index = {}

    def _save_index(self) -> None:
        """Save the runs index to disk."""
        try:
            self.index_file.write_text(json.dumps(self._runs_index, indent=2))
        except IOError as e:
            logger.error(f"Failed to save runs index: {e}")

    def _get_run_path(self, run_id: str) -> Path:
        """Get the file path for a run."""
        return self.runs_dir / f"run_{run_id}.json"

    def save_run(self, run: ExperimentRun) -> str:
        """Save an experiment run.

        Parameters
        ----------
        run : ExperimentRun
            The run to save

        Returns
        -------
        str
            Run ID
        """
        run_path = self._get_run_path(run.run_id)

        try:
            run_path.write_text(json.dumps(run.to_dict(), indent=2, default=str))
        except IOError as e:
            logger.error(f"Failed to save run: {e}")
            return run.run_id

        # Update index
        self._runs_index[run.run_id] = {
            "agent_type": run.agent_type,
            "model": run.model,
            "tool_bundle": run.tool_bundle,
            "status": run.status,
            "started": run.started,
            "ended": run.ended,
            "tool_call_count": run.tool_call_count,
            "total_duration_ms": run.total_duration_ms,
            "tokens_total": run.tokens_total,
            "tags": run.tags,
        }
        self._save_index()

        # Cleanup if over limit
        self._cleanup_if_needed()

        logger.debug(f"Saved run {run.run_id}")
        return run.run_id

    def load_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Load an experiment run.

        Parameters
        ----------
        run_id : str
            Run ID to load

        Returns
        -------
        ExperimentRun or None
            The run or None if not found
        """
        run_path = self._get_run_path(run_id)

        if not run_path.exists():
            logger.debug(f"Run not found: {run_id}")
            return None

        try:
            data = json.loads(run_path.read_text())
            return ExperimentRun.from_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load run {run_id}: {e}")
            return None

    def list_runs(
        self,
        agent_type: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List runs with optional filtering.

        Parameters
        ----------
        agent_type : str, optional
            Filter by agent type
        model : str, optional
            Filter by model
        status : str, optional
            Filter by status
        tags : list of str, optional
            Filter by tags (any match)
        limit : int
            Maximum runs to return

        Returns
        -------
        list of dict
            Run metadata sorted by start time (newest first)
        """
        runs = []

        for run_id, info in self._runs_index.items():
            # Apply filters
            if agent_type and info.get("agent_type") != agent_type:
                continue
            if model and info.get("model") != model:
                continue
            if status and info.get("status") != status:
                continue
            if tags:
                run_tags = set(info.get("tags", []))
                if not run_tags.intersection(set(tags)):
                    continue

            runs.append({"run_id": run_id, **info})

        # Sort by start time, newest first
        runs.sort(key=lambda r: r.get("started", ""), reverse=True)

        return runs[:limit]

    def delete_run(self, run_id: str) -> bool:
        """Delete a run.

        Parameters
        ----------
        run_id : str
            Run ID to delete

        Returns
        -------
        bool
            True if deleted
        """
        run_path = self._get_run_path(run_id)

        if run_path.exists():
            run_path.unlink()

        if run_id in self._runs_index:
            del self._runs_index[run_id]
            self._save_index()
            return True

        return False

    def _cleanup_if_needed(self) -> None:
        """Remove old runs if over limit."""
        if len(self._runs_index) <= self.max_runs:
            return

        # Sort by start time and remove oldest
        runs = list(self._runs_index.items())
        runs.sort(key=lambda x: x[1].get("started", ""))

        to_remove = len(runs) - self.max_runs
        for run_id, _ in runs[:to_remove]:
            self.delete_run(run_id)

    # Benchmark suite management

    def save_benchmark(self, suite: BenchmarkSuite) -> str:
        """Save a benchmark suite.

        Parameters
        ----------
        suite : BenchmarkSuite
            Benchmark suite to save

        Returns
        -------
        str
            Suite ID
        """
        suite_path = self.benchmarks_dir / f"benchmark_{suite.suite_id}.json"

        try:
            suite_path.write_text(json.dumps(suite.to_dict(), indent=2, default=str))
            logger.debug(f"Saved benchmark suite {suite.suite_id}")
        except IOError as e:
            logger.error(f"Failed to save benchmark: {e}")

        return suite.suite_id

    def load_benchmark(self, suite_id: str) -> Optional[BenchmarkSuite]:
        """Load a benchmark suite.

        Parameters
        ----------
        suite_id : str
            Suite ID to load

        Returns
        -------
        BenchmarkSuite or None
            The suite or None if not found
        """
        suite_path = self.benchmarks_dir / f"benchmark_{suite_id}.json"

        if not suite_path.exists():
            return None

        try:
            data = json.loads(suite_path.read_text())
            return BenchmarkSuite.from_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load benchmark {suite_id}: {e}")
            return None

    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """List all benchmark suites.

        Returns
        -------
        list of dict
            Benchmark metadata
        """
        benchmarks = []

        for path in self.benchmarks_dir.glob("benchmark_*.json"):
            try:
                data = json.loads(path.read_text())
                benchmarks.append({
                    "suite_id": data.get("suite_id"),
                    "name": data.get("name"),
                    "description": data.get("description"),
                    "created": data.get("created"),
                    "result_count": len(data.get("results", [])),
                })
            except (json.JSONDecodeError, IOError):
                continue

        benchmarks.sort(key=lambda b: b.get("created", ""), reverse=True)
        return benchmarks

    # Analytics

    def get_stats(
        self,
        agent_type: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregate statistics for runs.

        Parameters
        ----------
        agent_type : str, optional
            Filter by agent type
        model : str, optional
            Filter by model

        Returns
        -------
        dict
            Aggregate statistics
        """
        runs = self.list_runs(agent_type=agent_type, model=model, limit=10000)

        if not runs:
            return {
                "total_runs": 0,
                "completed": 0,
                "failed": 0,
            }

        completed = [r for r in runs if r.get("status") == "completed"]
        failed = [r for r in runs if r.get("status") == "failed"]

        durations = [r["total_duration_ms"] for r in completed if r.get("total_duration_ms")]
        tool_counts = [r["tool_call_count"] for r in completed if r.get("tool_call_count")]
        token_counts = [r["tokens_total"] for r in completed if r.get("tokens_total")]

        return {
            "total_runs": len(runs),
            "completed": len(completed),
            "failed": len(failed),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "avg_tool_calls": sum(tool_counts) / len(tool_counts) if tool_counts else 0,
            "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "total_tokens": sum(token_counts),
        }

    def export_runs(
        self,
        output_path: Union[str, Path],
        agent_type: Optional[str] = None,
        model: Optional[str] = None,
        format: str = "json",
    ) -> int:
        """Export runs to a file.

        Parameters
        ----------
        output_path : str or Path
            Output file path
        agent_type : str, optional
            Filter by agent type
        model : str, optional
            Filter by model
        format : str
            Export format ("json" or "csv")

        Returns
        -------
        int
            Number of runs exported
        """
        runs_meta = self.list_runs(agent_type=agent_type, model=model, limit=10000)

        if format == "json":
            # Full export with all details
            full_runs = []
            for meta in runs_meta:
                run = self.load_run(meta["run_id"])
                if run:
                    full_runs.append(run.to_dict())

            Path(output_path).write_text(json.dumps(full_runs, indent=2, default=str))

        elif format == "csv":
            import csv

            with open(output_path, "w", newline="") as f:
                if not runs_meta:
                    return 0

                writer = csv.DictWriter(f, fieldnames=list(runs_meta[0].keys()))
                writer.writeheader()
                writer.writerows(runs_meta)

        return len(runs_meta)


# Default experiment log instance
_default_log: Optional[ExperimentLog] = None


def get_default_log() -> ExperimentLog:
    """Get the default experiment log, creating it if needed."""
    global _default_log
    if _default_log is None:
        _default_log = ExperimentLog()
    return _default_log


def set_default_log(log: ExperimentLog) -> None:
    """Set the default experiment log instance."""
    global _default_log
    _default_log = log


def init_experiment_log(
    root_dir: Union[str, Path] = "./experiments",
    max_runs: int = 1000,
) -> ExperimentLog:
    """Initialize and set the default experiment log.

    Parameters
    ----------
    root_dir : str or Path
        Root directory for logs
    max_runs : int
        Maximum runs to retain

    Returns
    -------
    ExperimentLog
        The initialized log instance
    """
    log = ExperimentLog(root_dir, max_runs)
    set_default_log(log)
    return log
