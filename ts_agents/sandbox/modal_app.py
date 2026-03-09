"""Modal app for ts-agents sandbox execution from a source checkout.

This file defines a Modal Function that can execute ts-agents tools remotely.

Deploy:
    modal deploy -m ts_agents.sandbox.modal_app --env main --name ts-agents-sandbox

This module assumes a repository checkout because it builds the Modal image from
the local ``pyproject.toml`` and local ``ts_agents/`` source tree. It is not a
general installed-package deployment entrypoint.

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
PACKAGE_ROOT = REPO_ROOT / "ts_agents"
RESOURCES_ROOT = PACKAGE_ROOT / "resources"


app = modal.App("ts-agents-sandbox")


def _has_source_checkout() -> bool:
    return PYPROJECT.is_file() and PACKAGE_ROOT.is_dir() and RESOURCES_ROOT.is_dir()


def _require_source_checkout() -> None:
    if not _has_source_checkout():
        raise RuntimeError(
            "ts_agents.sandbox.modal_app must be executed from a source checkout "
            "that contains pyproject.toml and the ts_agents package tree."
        )


def _build_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.11")
    if not _has_source_checkout():
        return image
    return (
        image
        .pip_install_from_pyproject(str(PYPROJECT))
        .add_local_python_source("ts_agents")
        .add_local_dir(str(RESOURCES_ROOT), remote_path="/root/ts_agents/resources")
    )

# Build an image that:
#  - installs dependencies from pyproject.toml
#  - ships the canonical `ts_agents/` package
#  - includes non-python runtime resources under ts_agents/resources
image = _build_image()


@app.function(image=image, timeout=600)
def run_tool(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool request and return an ExecutionResult dict."""
    _require_source_checkout()

    # Import inside the function to ensure the packaged code is available.
    from ts_agents.tools.executor import ExecutionContext, SandboxMode, execute_tool

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
        modal run ts_agents/sandbox/modal_app.py --tool-name describe_series --series "[1,2,3]"
    """
    _require_source_checkout()
    payload = {"tool_name": tool_name, "kwargs": kwargs}
    print(run_tool.remote(payload))
