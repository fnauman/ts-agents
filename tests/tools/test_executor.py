"""Tests for tool executor serialization and context handling."""

import json
import numpy as np

from src.core.base import PeakResult
from src.tools.executor import ExecutionContext, ExecutionResult, ExecutionStatus, SandboxMode
from src.tools.results import DecompositionResult as ToolDecompositionResult


def test_execution_context_coerces_string():
    ctx = ExecutionContext(sandbox_mode="docker")
    assert ctx.sandbox_mode == SandboxMode.DOCKER

    ctx = ExecutionContext(sandbox_mode="LOCAL")
    assert ctx.sandbox_mode == SandboxMode.LOCAL


def test_execution_result_serializes_analysis_result():
    result = PeakResult(
        method="test",
        peak_indices=np.array([1, 2]),
        peak_values=np.array([0.1, 0.2]),
        count=2,
    )

    exec_result = ExecutionResult(status=ExecutionStatus.SUCCESS, result=result)
    payload = exec_result.to_dict()

    assert payload["result"]["peak_indices"] == [1, 2]
    assert payload["result"]["peak_values"] == [0.1, 0.2]
    json.dumps(payload)


def test_execution_result_serializes_tool_result():
    result = ToolDecompositionResult(
        trend=[1.0],
        seasonal=[],
        residual=[0.0],
        period=1,
        method="stl",
    )

    exec_result = ExecutionResult(status=ExecutionStatus.SUCCESS, result=result)
    payload = exec_result.to_dict()

    assert payload["result"]["trend"] == [1.0]
    assert payload["result"]["period"] == 1
    json.dumps(payload)
