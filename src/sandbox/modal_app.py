"""Modal app for ts-agents sandbox execution.

This file defines a Modal Function that can execute ts-agents tools remotely.

Deploy:
    modal deploy src/sandbox/modal_app.py

Then configure ts-agents to use Modal as its sandbox backend by setting:
    TS_AGENTS_SANDBOX_MODE=modal
    TS_AGENTS_MODAL_APP=ts-agents-sandbox
    TS_AGENTS_MODAL_FUNCTION=run_tool

Or pass `ExecutionContext(sandbox_mode="modal", modal_app_name=..., modal_function_name=...)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import modal


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


app = modal.App("ts-agents-sandbox")

# Build an image that:
#  - installs dependencies from pyproject.toml
#  - ships the local `src/` python package
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject(str(PYPROJECT))
    .add_local_python_source("src")
)


@app.function(image=image, timeout=600)
def run_tool(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool request and return an ExecutionResult dict."""

    # Import inside the function to ensure the packaged code is available.
    from src.tools.executor import ExecutionContext, SandboxMode, execute_tool

    tool_name = request.get("tool_name")
    params = request.get("kwargs") or {}
    ctx_in = request.get("context") or {}

    allowed = {
        k: ctx_in.get(k)
        for k in [
            "timeout_seconds",
            "memory_mb",
            "disk_mb",
            "working_dir",
        ]
        if k in ctx_in
    }

    ctx = ExecutionContext(
        sandbox_mode=SandboxMode.LOCAL,
        user_approved=True,
        **allowed,
    )

    result = execute_tool(tool_name, params, ctx)
    return result.to_dict()


@app.local_entrypoint()
def main(tool_name: str = "describe_series", **kwargs: Any):
    """Local smoke test.

    Example:
        modal run src/sandbox/modal_app.py --tool-name describe_series --series "[1,2,3]"
    """
    payload = {"tool_name": tool_name, "kwargs": kwargs}
    print(run_tool.remote(payload))
