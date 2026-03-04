"""Persistence layer for caching, sessions, and experiment logging.

This module provides comprehensive persistence capabilities:

- **ResultsCache**: Cache analysis results to avoid recomputation
- **SessionStore**: Persist Gradio UI sessions across page reloads
- **ExperimentLog**: Log agent interactions for analysis and benchmarking

Examples
--------
>>> from ts_agents.persistence import ResultsCache, init_cache
>>>
>>> # Initialize the results cache
>>> cache = init_cache("./results")
>>>
>>> # Get or compute a result
>>> result = cache.get_or_compute(
...     run_id="Re200Rm200",
...     variable="bx001_real",
...     method="stl_decompose",
...     params={"period": 150},
...     compute_fn=lambda: stl_decompose(series, period=150)
... )

>>> from ts_agents.persistence import SessionStore, SessionState
>>>
>>> # Save and restore a Gradio session
>>> store = SessionStore("./sessions")
>>> state = SessionState()
>>> state.current_run_id = "Re200Rm200"
>>> store.save(state)
>>> restored = store.load(state.session_id)

>>> from ts_agents.persistence import ExperimentLog, ExperimentRun, ToolCall
>>>
>>> # Log an agent experiment
>>> log = ExperimentLog("./experiments")
>>> run = ExperimentRun(agent_type="simple", query="Count the peaks")
>>> run.add_tool_call(ToolCall(tool_name="detect_peaks"))
>>> run.complete("Found 42 peaks")
>>> log.save_run(run)
"""

from .results_cache import (
    ResultsCache,
    cached,
    get_default_cache,
    set_default_cache,
    init_cache,
)

from .session_store import (
    SessionState,
    SessionStore,
    get_default_store,
    set_default_store,
    init_session_store,
)

from .experiment_log import (
    ExperimentStatus,
    ToolCall,
    ExperimentRun,
    BenchmarkSuite,
    ExperimentLog,
    get_default_log,
    set_default_log,
    init_experiment_log,
)

__all__ = [
    # Results cache
    "ResultsCache",
    "cached",
    "get_default_cache",
    "set_default_cache",
    "init_cache",
    # Session store
    "SessionState",
    "SessionStore",
    "get_default_store",
    "set_default_store",
    "init_session_store",
    # Experiment log
    "ExperimentStatus",
    "ToolCall",
    "ExperimentRun",
    "BenchmarkSuite",
    "ExperimentLog",
    "get_default_log",
    "set_default_log",
    "init_experiment_log",
]
