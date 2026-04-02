"""Sandbox tool runner.

This module is designed to be the *single* entrypoint that runs inside an
isolated execution environment.

It reads a JSON "tool request" from disk, executes the requested ts-agents tool
*locally inside the sandbox*, and writes a serialized ExecutionResult JSON back
to disk.

Why a file-based interface?
-------------------------
Docker/Daytona/Modal integrations all have slightly different ways to pass data
in/out. A small file-based contract is the most interoperable option:

Input schema (JSON)
-------------------
{
  "tool_name": "forecast_theta_with_data",
  "kwargs": {"run_id": "Re200Rm200", "variable_name": "y", "horizon": 50},
  "context": {"timeout_seconds": 120, "memory_mb": 2048}
}

Output schema (JSON)
--------------------
ExecutionResult.to_dict()
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ts_agents.cli.output import dump_json


def run_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool request payload and return a serialized ExecutionResult."""

    from ts_agents.tools.executor import ExecutionContext, SandboxMode, execute_tool
    from ts_agents.workflows.executor import (
        execute_serialized_workflow_request,
        is_workflow_target,
    )

    tool_name = payload.get("tool_name")
    if not tool_name:
        raise ValueError("Missing required field: tool_name")

    kwargs = payload.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dict")

    context_in = payload.get("context") or {}
    if not isinstance(context_in, dict):
        raise TypeError("context must be a dict")

    # Force LOCAL execution inside the sandbox to avoid nested sandbox calls.
    # If the caller wants extra isolation, they should choose the sandbox type
    # at the *outer* layer (Docker/Daytona/Modal).
    context = ExecutionContext(
        sandbox_mode=SandboxMode.LOCAL,
        user_approved=True,
        **{k: v for k, v in context_in.items() if k in {
            "timeout_seconds",
            "memory_mb",
            "disk_mb",
            "working_dir",
            "environment",
        }},
    )

    if is_workflow_target(tool_name):
        workflow_name = tool_name.split(":", 1)[1]
        result = execute_serialized_workflow_request(
            workflow_name=workflow_name,
            kwargs=kwargs,
            context=context,
        )
    else:
        result = execute_tool(tool_name, kwargs, context=context)
    return result.to_dict()


def run_request_file(input_path: str | Path, output_path: str | Path) -> Path:
    """Load a request JSON file, execute it, and write the response JSON."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    payload = json.loads(input_path.read_text())
    response = run_request(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_json(response))
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a ts-agents tool request inside a sandbox")
    parser.add_argument("--input", "--in", dest="input", required=True, help="Path to input JSON")
    parser.add_argument("--output", "--out", dest="output", required=True, help="Path to output JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    run_request_file(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
