"""Tool Execution Abstraction Layer.

This module provides an abstraction for tool execution that can route
calls to local execution, Docker containers, or other sandboxed environments.

The executor is a key architectural component for Phase 4+ sandbox support.

Example usage:
    >>> from ts_agents.tools.executor import ToolExecutor, ExecutionContext, SandboxMode
    >>>
    >>> executor = ToolExecutor()
    >>> context = ExecutionContext(sandbox_mode=SandboxMode.LOCAL)
    >>> result = executor.execute("stl_decompose_with_data",
    ...     {"variable_name": "bx001_real", "unique_id": "Re200Rm200"},
    ...     context=context
    ... )
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .results import format_result, serialize_result

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """Execution environment modes."""
    LOCAL = "local"          # Direct local execution (no sandbox)
    DOCKER = "docker"        # Docker container sandbox
    DAYTONA = "daytona"      # Daytona cloud sandbox
    MODAL = "modal"          # Modal cloud execution
    SUBPROCESS = "subprocess"  # Local subprocess isolation


def _coerce_sandbox_mode(value: Union["SandboxMode", str, None]) -> "SandboxMode":
    """Coerce sandbox mode from string or enum to SandboxMode."""
    if value is None:
        return SandboxMode.LOCAL
    if isinstance(value, SandboxMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for mode in SandboxMode:
            if normalized in {mode.value, mode.name.lower()}:
                return mode
    raise ValueError(f"Invalid sandbox mode: {value}")


class ExecutionStatus(Enum):
    """Status of a tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """Context for tool execution.

    Attributes
    ----------
    sandbox_mode : SandboxMode | str
        Execution environment mode (enum or string like "local")
    timeout_seconds : int, optional
        Maximum execution time (defaults to tool metadata if None)
    memory_mb : int, optional
        Memory limit for sandboxed execution (defaults to tool metadata if None)
    disk_mb : int, optional
        Disk limit for sandboxed execution (defaults to tool metadata if None)
    working_dir : str, optional
        Working directory for execution
    environment : Dict[str, str]
        Additional environment variables
    user_approved : bool
        Whether user has approved this execution (for high-cost tools)
    """
    sandbox_mode: Union[SandboxMode, str] = SandboxMode.LOCAL
    timeout_seconds: Optional[int] = None
    memory_mb: Optional[int] = None
    disk_mb: Optional[int] = None
    working_dir: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    user_approved: bool = False

    # ------------------------------------------------------------------
    # Cross-sandbox security toggles
    # ------------------------------------------------------------------
    allow_network: bool = False

    # ------------------------------------------------------------------
    # Docker backend options
    # ------------------------------------------------------------------
    docker_image: Optional[str] = None
    docker_cpus: Optional[float] = None

    # ------------------------------------------------------------------
    # Daytona backend options
    # ------------------------------------------------------------------
    daytona_language: str = "python"
    daytona_ephemeral: bool = True
    daytona_name: Optional[str] = None
    # Use a Python-compatible Daytona snapshot by default.
    daytona_snapshot: Optional[str] = "daytonaio/sandbox:0.4.3"
    # Default bootstrap: clone/install repo before running tools.
    # Set to "" to disable clone when using a prebuilt snapshot image.
    daytona_repo_url: Optional[str] = "https://github.com/fnauman/ts-agents"
    daytona_repo_branch: Optional[str] = None
    daytona_repo_path: str = "workspace/ts-agents"
    daytona_git_username: Optional[str] = None
    daytona_git_password: Optional[str] = None
    daytona_install_editable: bool = True
    daytona_stream_logs: bool = True
    daytona_log_file: Optional[str] = None

    # ------------------------------------------------------------------
    # Modal backend options
    # ------------------------------------------------------------------
    modal_app_name: Optional[str] = None
    modal_function_name: Optional[str] = None
    modal_environment: Optional[str] = None
    modal_stream_logs: bool = True
    modal_log_file: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize sandbox mode inputs."""
        self.sandbox_mode = _coerce_sandbox_mode(self.sandbox_mode)


@dataclass
class ExecutionResult:
    """Result of a tool execution.

    Attributes
    ----------
    status : ExecutionStatus
        Execution status
    result : Any
        The structured result from the tool (if successful)
    formatted_output : str
        Formatted output string for LLM consumption
    error : ToolError, optional
        Error information if failed
    duration_ms : float
        Execution duration in milliseconds
    metadata : Dict[str, Any]
        Additional execution metadata
    """
    status: ExecutionStatus
    result: Optional[Any] = None
    formatted_output: str = ""
    error: Optional["ToolError"] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for IPC/serialization."""
        return {
            "status": self.status.value,
            "result": serialize_result(self.result),
            "formatted_output": self.formatted_output,
            "error": self.error.to_dict() if self.error else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """Create from dictionary."""
        return cls(
            status=ExecutionStatus(data["status"]),
            result=data.get("result"),
            formatted_output=data.get("formatted_output", ""),
            error=ToolError.from_dict(data["error"]) if data.get("error") else None,
            duration_ms=data.get("duration_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


class ToolErrorCode(Enum):
    """Standard error codes for tool execution."""
    UNKNOWN = "unknown"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    DATA_ERROR = "data_error"
    COMPUTATION_ERROR = "computation_error"
    DEPENDENCY_ERROR = "dependency_error"
    SERIALIZATION_ERROR = "serialization_error"


@dataclass
class ToolError(Exception):
    """Structured error for tool execution.

    This replaces the generic exception strings with structured errors
    that agents can parse and potentially recover from.

    Attributes
    ----------
    code : ToolErrorCode
        Error category code
    message : str
        Human-readable error message
    recoverable : bool
        Whether the error might be recoverable (e.g., retry with different params)
    details : Dict[str, Any]
        Additional error details
    tool_name : str, optional
        Name of the tool that failed
    """
    code: ToolErrorCode
    message: str
    recoverable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    tool_name: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for IPC/serialization."""
        return {
            "code": self.code.value,
            "message": self.message,
            "recoverable": self.recoverable,
            "details": self.details,
            "tool_name": self.tool_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolError":
        """Create from dictionary."""
        return cls(
            code=ToolErrorCode(data.get("code", "unknown")),
            message=data.get("message", "Unknown error"),
            recoverable=data.get("recoverable", False),
            details=data.get("details", {}),
            tool_name=data.get("tool_name"),
        )

    @classmethod
    def from_exception(cls, exc: Exception, tool_name: Optional[str] = None) -> "ToolError":
        """Create ToolError from a generic exception."""
        if isinstance(exc, ToolError):
            return exc
        # Map common exception types to error codes
        exc_type = type(exc).__name__

        code_mapping = {
            "ValueError": ToolErrorCode.VALIDATION_ERROR,
            "KeyError": ToolErrorCode.NOT_FOUND,
            "FileNotFoundError": ToolErrorCode.NOT_FOUND,
            "PermissionError": ToolErrorCode.PERMISSION_DENIED,
            "TimeoutError": ToolErrorCode.TIMEOUT,
            "MemoryError": ToolErrorCode.RESOURCE_EXHAUSTED,
            "ImportError": ToolErrorCode.DEPENDENCY_ERROR,
            "ModuleNotFoundError": ToolErrorCode.DEPENDENCY_ERROR,
        }

        code = code_mapping.get(exc_type, ToolErrorCode.UNKNOWN)

        # Determine if recoverable
        recoverable = code in {
            ToolErrorCode.VALIDATION_ERROR,
            ToolErrorCode.TIMEOUT,
            ToolErrorCode.DATA_ERROR,
        }

        return cls(
            code=code,
            message=str(exc),
            recoverable=recoverable,
            details={"exception_type": exc_type},
            tool_name=tool_name,
        )


class ExecutorBackend(ABC):
    """Abstract base class for execution backends."""

    @abstractmethod
    def execute(
        self,
        tool_name: str,
        func: Callable,
        params: Dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute a tool function.

        Parameters
        ----------
        tool_name : str
            Name of the tool being executed
        func : Callable
            The tool function to execute
        params : Dict[str, Any]
            Parameters to pass to the tool
        context : ExecutionContext
            Execution context with resource limits

        Returns
        -------
        ExecutionResult
            The execution result
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class LocalBackend(ExecutorBackend):
    """Local direct execution backend (no sandboxing)."""

    def execute(
        self,
        tool_name: str,
        func: Callable,
        params: Dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute a tool function locally."""
        start_time = time.time()

        # Apply working directory + environment variables (best-effort).
        old_cwd: Optional[str] = None
        old_env: Dict[str, Optional[str]] = {}

        try:
            if context.working_dir:
                old_cwd = os.getcwd()
                os.chdir(context.working_dir)

            for k, v in (context.environment or {}).items():
                old_env[k] = os.environ.get(k)
                os.environ[k] = str(v)

            result = func(**params)
            duration_ms = (time.time() - start_time) * 1000

            # Check if result is already structured or needs formatting
            formatted_output = self._format_result(result)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                result=result,
                formatted_output=formatted_output,
                duration_ms=duration_ms,
                metadata={
                    "tool_name": tool_name,
                    "backend": "local",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error = ToolError.from_exception(e, tool_name=tool_name)

            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=error,
                formatted_output=str(error),
                duration_ms=duration_ms,
                metadata={
                    "tool_name": tool_name,
                    "backend": "local",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        finally:
            # Restore env + cwd
            for k, prev in old_env.items():
                if prev is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev
            if old_cwd is not None:
                os.chdir(old_cwd)

    def _format_result(self, result: Any) -> str:
        """Format a result for LLM consumption."""
        return format_result(result)

    def _format_dict_result(self, result: Dict[str, Any]) -> str:
        """Format a structured dict result."""
        # Handle new structured format with visualization
        parts = []

        if "summary" in result:
            parts.append(result["summary"])
        elif "result" in result:
            parts.append(str(result["result"]))
        else:
            # Format key-value pairs
            for key, value in result.items():
                if key not in ("visualization", "image_data", "metadata"):
                    parts.append(f"{key}: {value}")

        # Handle visualization if present
        if "visualization" in result:
            viz = result["visualization"]
            if isinstance(viz, dict) and "data" in viz:
                parts.append(f"\n[IMAGE_DATA:{viz['data']}]")

        return "\n".join(parts) if parts else str(result)

    def is_available(self) -> bool:
        """Local backend is always available."""
        return True


class DockerBackend(ExecutorBackend):
    """Docker container execution backend.

    This backend executes tools in an isolated Docker container. It uses the
    file-based runner contract in :mod:`ts_agents.sandbox.runner`.

    Requirements
    ------------
    - Docker CLI available on the host.
    - A Docker image that contains the ts-agents code + dependencies.
      Default: ``ts-agents-sandbox:latest`` (override via
      ``ExecutionContext.docker_image`` or ``TS_AGENTS_DOCKER_IMAGE``).
    """

    def __init__(self, image: str = "ts-agents-sandbox:latest"):
        self.image = image
        self._docker_available: Optional[bool] = None

    def execute(
        self,
        tool_name: str,
        func: Callable,  # unused (kept for interface compatibility)
        params: Dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        start_time = time.time()

        if not self.is_available():
            err = ToolError(
                code=ToolErrorCode.DEPENDENCY_ERROR,
                message="Docker CLI is not available. Install Docker or use a different sandbox_mode.",
                recoverable=True,
                tool_name=tool_name,
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=err,
                formatted_output=str(err),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"tool_name": tool_name, "backend": "docker"},
            )

        image = (
            context.docker_image
            or os.environ.get("TS_AGENTS_DOCKER_IMAGE")
            or self.image
        )
        timeout_s = context.timeout_seconds or int(os.environ.get("TS_AGENTS_DOCKER_TIMEOUT", "300"))

        # Resource defaults (best-effort; Docker may enforce its own limits)
        memory_mb = context.memory_mb or int(os.environ.get("TS_AGENTS_DOCKER_MEMORY_MB", "2048"))
        cpus = context.docker_cpus
        if cpus is None:
            try:
                cpus = float(os.environ.get("TS_AGENTS_DOCKER_CPUS", "2"))
            except Exception:
                cpus = 2.0

        request_payload = {
            "tool_name": tool_name,
            "kwargs": params,
            "context": {
                "timeout_seconds": context.timeout_seconds,
                "memory_mb": context.memory_mb,
                "disk_mb": context.disk_mb,
                "working_dir": context.working_dir,
            },
        }

        try:
            with tempfile.TemporaryDirectory(prefix="ts_agents_docker_") as td:
                io_dir = Path(td)
                req_path = io_dir / "request.json"
                resp_path = io_dir / "response.json"
                req_path.write_text(json.dumps(request_payload, indent=2, default=str))

                cmd: List[str] = [
                    "docker",
                    "run",
                    "--rm",
                    "--security-opt=no-new-privileges",
                    "--pids-limit",
                    "512",
                    "--read-only",
                ]

                if not context.allow_network:
                    cmd += ["--network", "none"]

                if memory_mb:
                    cmd += ["--memory", f"{int(memory_mb)}m"]
                if cpus:
                    cmd += ["--cpus", str(cpus)]

                # I/O volume
                cmd += ["-v", f"{io_dir}:/io"]

                # Optional: mount TS_AGENTS_DATA_DIR for file-based workflows.
                host_data_dir = os.environ.get("TS_AGENTS_DATA_DIR")
                if host_data_dir and os.path.isdir(host_data_dir):
                    cmd += ["-v", f"{host_data_dir}:{host_data_dir}:ro", "-e", f"TS_AGENTS_DATA_DIR={host_data_dir}"]

                # Pass user-provided env vars
                for k, v in (context.environment or {}).items():
                    cmd += ["-e", f"{k}={v}"]

                # Run the sandbox runner inside the container
                cmd += [
                    image,
                    "python",
                    "-m",
                    "ts_agents.sandbox.runner",
                    "--input",
                    "/io/request.json",
                    "--output",
                    "/io/response.json",
                ]

                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )

                if completed.returncode != 0:
                    err = ToolError(
                        code=ToolErrorCode.COMPUTATION_ERROR,
                        message=f"Docker execution failed (exit code {completed.returncode}).",
                        recoverable=True,
                        tool_name=tool_name,
                        details={
                            "stdout": completed.stdout[-4000:],
                            "stderr": completed.stderr[-4000:],
                            "image": image,
                        },
                    )
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=err,
                        formatted_output=str(err),
                        duration_ms=(time.time() - start_time) * 1000,
                        metadata={"tool_name": tool_name, "backend": "docker", "image": image},
                    )

                if not resp_path.exists():
                    err = ToolError(
                        code=ToolErrorCode.SERIALIZATION_ERROR,
                        message="Docker runner did not produce a response.json file.",
                        recoverable=True,
                        tool_name=tool_name,
                        details={"stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]},
                    )
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=err,
                        formatted_output=str(err),
                        duration_ms=(time.time() - start_time) * 1000,
                        metadata={"tool_name": tool_name, "backend": "docker", "image": image},
                    )

                response_dict = json.loads(resp_path.read_text())
                result = ExecutionResult.from_dict(response_dict)

                # Augment metadata (preserve sandbox-side metadata too)
                result.metadata = {**(result.metadata or {}), "backend": "docker", "image": image}
                return result

        except subprocess.TimeoutExpired:
            err = ToolError(
                code=ToolErrorCode.TIMEOUT,
                message=f"Docker execution timed out after {timeout_s} seconds.",
                recoverable=True,
                tool_name=tool_name,
            )
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=err,
                formatted_output=str(err),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"tool_name": tool_name, "backend": "docker", "image": image},
            )

        except Exception as e:
            err = ToolError.from_exception(e, tool_name=tool_name)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=err,
                formatted_output=str(err),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"tool_name": tool_name, "backend": "docker", "image": image},
            )

    def is_available(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=5,
            )
            self._docker_available = result.returncode == 0
        except Exception:
            self._docker_available = False

        return self._docker_available


class SubprocessBackend(ExecutorBackend):
    """Subprocess execution backend for local isolation.

    This backend executes tools in separate Python subprocesses
    for basic isolation without Docker overhead.
    """

    def execute(
        self,
        tool_name: str,
        func: Callable,
        params: Dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute a tool function in a separate Python process.

        This provides basic isolation (separate interpreter + address space)
        without requiring Docker.
        """
        start_time = time.time()
        timeout_s = context.timeout_seconds or int(os.environ.get("TS_AGENTS_SUBPROCESS_TIMEOUT", "300"))

        request_payload = {
            "tool_name": tool_name,
            "kwargs": params,
            "context": {
                "timeout_seconds": context.timeout_seconds,
                "memory_mb": context.memory_mb,
                "disk_mb": context.disk_mb,
                "working_dir": context.working_dir,
            },
        }

        try:
            with tempfile.TemporaryDirectory(prefix="ts_agents_subprocess_") as td:
                io_dir = Path(td)
                req_path = io_dir / "request.json"
                resp_path = io_dir / "response.json"
                req_path.write_text(json.dumps(request_payload, indent=2, default=str))

                env = os.environ.copy()
                for k, v in (context.environment or {}).items():
                    env[k] = str(v)

                cmd = [
                    sys.executable,
                    "-m",
                    "ts_agents.sandbox.runner",
                    "--input",
                    str(req_path),
                    "--output",
                    str(resp_path),
                ]

                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=context.working_dir or None,
                    env=env,
                )

                if completed.returncode != 0:
                    err = ToolError(
                        code=ToolErrorCode.COMPUTATION_ERROR,
                        message=f"Subprocess execution failed (exit code {completed.returncode}).",
                        recoverable=True,
                        tool_name=tool_name,
                        details={
                            "stdout": completed.stdout[-4000:],
                            "stderr": completed.stderr[-4000:],
                        },
                    )
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=err,
                        formatted_output=str(err),
                        duration_ms=(time.time() - start_time) * 1000,
                        metadata={"tool_name": tool_name, "backend": "subprocess"},
                    )

                if not resp_path.exists():
                    err = ToolError(
                        code=ToolErrorCode.SERIALIZATION_ERROR,
                        message="Subprocess runner did not produce a response.json file.",
                        recoverable=True,
                        tool_name=tool_name,
                        details={"stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]},
                    )
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=err,
                        formatted_output=str(err),
                        duration_ms=(time.time() - start_time) * 1000,
                        metadata={"tool_name": tool_name, "backend": "subprocess"},
                    )

                response_dict = json.loads(resp_path.read_text())
                result = ExecutionResult.from_dict(response_dict)
                result.metadata = {**(result.metadata or {}), "backend": "subprocess"}
                return result

        except subprocess.TimeoutExpired:
            err = ToolError(
                code=ToolErrorCode.TIMEOUT,
                message=f"Subprocess execution timed out after {timeout_s} seconds.",
                recoverable=True,
                tool_name=tool_name,
            )
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=err,
                formatted_output=str(err),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"tool_name": tool_name, "backend": "subprocess"},
            )

        except Exception as e:
            err = ToolError.from_exception(e, tool_name=tool_name)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=err,
                formatted_output=str(err),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"tool_name": tool_name, "backend": "subprocess"},
            )

    def is_available(self) -> bool:
        """Subprocess backend is always available."""
        return True


class DaytonaBackend(ExecutorBackend):
    """Daytona cloud sandbox execution backend.

    This backend requires the Daytona Python SDK (`pip install daytona`) and a
    configured API key.

    The implementation follows the Daytona docs patterns:
    - create a Sandbox via `Daytona().create(...)`
    - upload a JSON request via `sandbox.fs.upload_file(...)`
    - execute a command via `sandbox.process.exec(...)`
    - download the JSON response via `sandbox.fs.download_file(...)`
    """

    def __init__(self):
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import daytona  # noqa: F401

            self._available = True
        except Exception:
            self._available = False
        return self._available

    @staticmethod
    def _coerce_bool_env(var_name: str, default: bool) -> bool:
        raw = os.environ.get(var_name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _sanitize_stream_chunk(text: str) -> str:
        # Daytona PTY output may include control prefix bytes.
        return text.replace("\x01", "")

    def _run_command(
        self,
        sandbox: Any,
        command: str,
        *,
        cwd: Optional[str],
        timeout_s: int,
        env: Optional[Dict[str, str]] = None,
        stream_logs: bool = True,
        label: str = "daytona",
        log_file_handle: Optional[Any] = None,
    ) -> tuple[int, str]:
        """Execute a command inside Daytona and optionally stream output."""
        def _emit(chunk: str) -> None:
            line = f"[{label}] {chunk}"
            print(line, end="", file=sys.stderr, flush=True)
            if log_file_handle is not None:
                log_file_handle.write(line)
                if not line.endswith("\n"):
                    log_file_handle.write("\n")
                log_file_handle.flush()

        if not stream_logs:
            resp = sandbox.process.exec(command, cwd=cwd, timeout=timeout_s, env=env or None)
            exit_code = getattr(resp, "exit_code", 0)
            output = getattr(resp, "result", "") or ""
            if log_file_handle is not None and output:
                _emit(str(output))
            return int(exit_code or 0), str(output)

        pty = None
        chunks: List[str] = []
        session_id = f"ts-agents-{uuid.uuid4().hex[:8]}"
        try:
            pty = sandbox.process.create_pty_session(session_id, cwd=cwd, envs=env or None)
            pty.wait_for_connection(timeout=10.0)
            pty.send_input(command + "\n")
            pty.send_input("exit\n")

            def _on_data(data: bytes) -> None:
                chunk = self._sanitize_stream_chunk(data.decode("utf-8", errors="ignore"))
                chunks.append(chunk)
                _emit(chunk)

            result = pty.wait(on_data=_on_data, timeout=float(timeout_s))
            exit_code = result.exit_code
            if exit_code is None:
                # Timed out or disconnected before receiving terminal exit code.
                if result.error:
                    chunks.append(result.error)
                return 124, "".join(chunks)
            return int(exit_code), "".join(chunks)
        except Exception:
            # Best-effort fallback when PTY APIs are unavailable.
            resp = sandbox.process.exec(command, cwd=cwd, timeout=timeout_s, env=env or None)
            exit_code = getattr(resp, "exit_code", 0)
            output = getattr(resp, "result", "") or ""
            return int(exit_code or 0), str(output)
        finally:
            if pty is not None:
                try:
                    pty.disconnect()
                except Exception:
                    pass

    def execute(
        self,
        tool_name: str,
        func: Callable,  # unused
        params: Dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        start_time = time.time()

        if not self.is_available():
            err = ToolError(
                code=ToolErrorCode.DEPENDENCY_ERROR,
                message="Daytona SDK is not available. Install it with 'pip install daytona' or use another sandbox_mode.",
                recoverable=True,
                tool_name=tool_name,
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=err,
                formatted_output=str(err),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"tool_name": tool_name, "backend": "daytona"},
            )

        timeout_s = context.timeout_seconds or int(os.environ.get("TS_AGENTS_DAYTONA_TIMEOUT", "300"))
        stream_logs = self._coerce_bool_env("TS_AGENTS_DAYTONA_STREAM", context.daytona_stream_logs)
        daytona_log_file = os.environ.get("TS_AGENTS_DAYTONA_LOG_FILE", context.daytona_log_file or "").strip() or None
        daytona_log_handle = None
        if daytona_log_file:
            log_path = Path(daytona_log_file).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            daytona_log_handle = log_path.open("a", encoding="utf-8")

        request_payload = {
            "tool_name": tool_name,
            "kwargs": params,
            "context": {
                "timeout_seconds": context.timeout_seconds,
                "memory_mb": context.memory_mb,
                "disk_mb": context.disk_mb,
                "working_dir": context.working_dir,
            },
        }

        sandbox = None
        try:
            from daytona import Daytona, CreateSandboxFromSnapshotParams

            repo_path = (context.daytona_repo_path or "workspace/ts-agents").rstrip("/")
            if not repo_path:
                repo_path = "workspace/ts-agents"

            daytona = Daytona()
            daytona_snapshot = os.environ.get("TS_AGENTS_DAYTONA_SNAPSHOT", context.daytona_snapshot or "").strip() or None
            create_params = CreateSandboxFromSnapshotParams(
                language=context.daytona_language or "python",
                name=context.daytona_name,
                ephemeral=bool(context.daytona_ephemeral),
                snapshot=daytona_snapshot,
            )
            sandbox = daytona.create(create_params)

            # Optional bootstrap by cloning a repo and installing dependencies.
            if context.daytona_repo_url:
                try:
                    # Prefer shell clone for better diagnostics and streamable logs.
                    if context.daytona_git_username or context.daytona_git_password:
                        # Preserve explicit credential fields for private repositories.
                        sandbox.git.clone(
                            context.daytona_repo_url,
                            path=repo_path,
                            branch=context.daytona_repo_branch,
                            username=context.daytona_git_username,
                            password=context.daytona_git_password,
                        )
                    else:
                        branch_part = ""
                        if context.daytona_repo_branch:
                            branch_part = f"--branch {shlex.quote(context.daytona_repo_branch)} --single-branch "
                        clone_cmd = (
                            f"rm -rf {shlex.quote(repo_path)} && "
                            f"git clone {branch_part}{shlex.quote(context.daytona_repo_url)} {shlex.quote(repo_path)}"
                        )
                        clone_exit, clone_output = self._run_command(
                            sandbox,
                            clone_cmd,
                            cwd=None,
                            timeout_s=timeout_s,
                            stream_logs=stream_logs,
                            label="daytona-bootstrap",
                            log_file_handle=daytona_log_handle,
                        )
                        if clone_exit != 0:
                            err = ToolError(
                                code=ToolErrorCode.DEPENDENCY_ERROR,
                                message=f"Daytona bootstrap clone failed (exit_code={clone_exit}).",
                                recoverable=True,
                                tool_name=tool_name,
                                details={"result": clone_output},
                            )
                            return ExecutionResult(
                                status=ExecutionStatus.FAILED,
                                error=err,
                                formatted_output=str(err),
                                duration_ms=(time.time() - start_time) * 1000,
                                metadata={"tool_name": tool_name, "backend": "daytona"},
                            )

                    if context.daytona_install_editable:
                        install_cmd = (
                            "bash -lc "
                            + shlex.quote(
                                "python -m venv .ts-agents-venv && "
                                ". .ts-agents-venv/bin/activate && "
                                "python -m pip install -e ."
                            )
                        )
                        install_exit, install_output = self._run_command(
                            sandbox,
                            install_cmd,
                            cwd=repo_path,
                            timeout_s=timeout_s,
                            stream_logs=stream_logs,
                            label="daytona-install",
                            log_file_handle=daytona_log_handle,
                        )
                        if install_exit != 0:
                            err = ToolError(
                                code=ToolErrorCode.DEPENDENCY_ERROR,
                                message=f"Daytona bootstrap install failed (exit_code={install_exit}).",
                                recoverable=True,
                                tool_name=tool_name,
                                details={"result": install_output},
                            )
                            return ExecutionResult(
                                status=ExecutionStatus.FAILED,
                                error=err,
                                formatted_output=str(err),
                                duration_ms=(time.time() - start_time) * 1000,
                                metadata={"tool_name": tool_name, "backend": "daytona"},
                            )
                except Exception as e:
                    err = ToolError(
                        code=ToolErrorCode.DEPENDENCY_ERROR,
                        message=f"Daytona bootstrap clone/install failed: {e!r}",
                        recoverable=True,
                        tool_name=tool_name,
                    )
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=err,
                        formatted_output=str(err),
                        duration_ms=(time.time() - start_time) * 1000,
                        metadata={"tool_name": tool_name, "backend": "daytona"},
                    )

            # Keep request/response files inside repo_path to avoid cwd/path ambiguity.
            remote_dir = f"{repo_path}/.ts_agents_io"
            try:
                sandbox.fs.create_folder(remote_dir, "755")
            except Exception:
                # Folder may already exist; ignore.
                pass

            req_remote = f"{remote_dir}/request.json"
            resp_remote = f"{remote_dir}/response.json"
            req_runner = ".ts_agents_io/request.json"
            resp_runner = ".ts_agents_io/response.json"

            sandbox.fs.upload_file(
                json.dumps(request_payload, indent=2, default=str).encode("utf-8"),
                req_remote,
            )

            env = {k: str(v) for k, v in (context.environment or {}).items()}
            runner_python = "python"
            if context.daytona_install_editable and context.daytona_repo_url:
                runner_python = ".ts-agents-venv/bin/python"
            cmd = (
                f"{runner_python} -m ts_agents.sandbox.runner --input {req_runner} --output {resp_runner}"
            )
            run_exit, run_output = self._run_command(
                sandbox,
                cmd,
                cwd=repo_path,
                timeout_s=timeout_s,
                env=env or None,
                stream_logs=stream_logs,
                label="daytona-run",
                log_file_handle=daytona_log_handle,
            )

            # Daytona returns an object with exit_code/result.
            if run_exit != 0:
                err = ToolError(
                    code=ToolErrorCode.COMPUTATION_ERROR,
                    message=f"Daytona process execution failed (exit_code={run_exit}).",
                    recoverable=True,
                    tool_name=tool_name,
                    details={"result": run_output},
                )
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=err,
                    formatted_output=str(err),
                    duration_ms=(time.time() - start_time) * 1000,
                    metadata={"tool_name": tool_name, "backend": "daytona"},
                )

            raw = sandbox.fs.download_file(resp_remote)
            response_dict = json.loads(raw.decode("utf-8"))
            result = ExecutionResult.from_dict(response_dict)
            result.metadata = {**(result.metadata or {}), "backend": "daytona"}
            return result

        except Exception as e:
            err = ToolError.from_exception(e, tool_name=tool_name)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=err,
                formatted_output=str(err),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"tool_name": tool_name, "backend": "daytona"},
            )

        finally:
            # Always clean up ephemeral sandboxes (best-effort).
            if sandbox is not None and context.daytona_ephemeral:
                try:
                    sandbox.delete()
                except Exception:
                    pass
            if daytona_log_handle is not None:
                try:
                    daytona_log_handle.close()
                except Exception:
                    pass


class ModalBackend(ExecutorBackend):
    """Modal cloud execution backend.

    Requires the `modal` Python package and a deployed Modal function that
    accepts a tool request dict and returns an ExecutionResult dict.
    """

    def __init__(self, app_name: str = "ts-agents-sandbox", function_name: str = "run_tool"):
        self.app_name = app_name
        self.function_name = function_name
        self._available: Optional[bool] = None

    @staticmethod
    def _coerce_bool_env(var_name: str, default: bool) -> bool:
        raw = os.environ.get(var_name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _open_log_file(path_value: Optional[str]) -> Optional[Any]:
        if not path_value:
            return None
        path = Path(path_value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8")

    @staticmethod
    def _emit_log(line: str, log_file_handle: Optional[Any]) -> None:
        print(line, end="", file=sys.stderr, flush=True)
        if log_file_handle is not None:
            log_file_handle.write(line)
            if not line.endswith("\n"):
                log_file_handle.write("\n")
            log_file_handle.flush()

    def _start_modal_log_stream(
        self,
        *,
        app_name: str,
        environment_name: Optional[str],
        log_file_handle: Optional[Any],
        timestamps: bool,
    ) -> tuple[Optional[subprocess.Popen], Optional[threading.Thread]]:
        if shutil.which("modal") is None:
            return None, None
        cmd = ["modal", "app", "logs", app_name]
        if environment_name:
            cmd.extend(["--env", environment_name])
        if timestamps:
            cmd.append("--timestamps")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _reader() -> None:
            if proc.stdout is None:
                return
            try:
                for line in proc.stdout:
                    self._emit_log(f"[modal-log] {line}", log_file_handle)
            except Exception:
                pass

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        return proc, thread

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import modal  # noqa: F401

            self._available = True
        except Exception:
            self._available = False
        return self._available

    def execute(
        self,
        tool_name: str,
        func: Callable,  # unused
        params: Dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        start_time = time.time()

        if not self.is_available():
            err = ToolError(
                code=ToolErrorCode.DEPENDENCY_ERROR,
                message="Modal SDK is not available. Install it with 'pip install modal' or use another sandbox_mode.",
                recoverable=True,
                tool_name=tool_name,
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=err,
                formatted_output=str(err),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"tool_name": tool_name, "backend": "modal"},
            )

        modal_log_file = os.environ.get("TS_AGENTS_MODAL_LOG_FILE", context.modal_log_file or "").strip() or None
        modal_log_handle = self._open_log_file(modal_log_file)
        logs_proc: Optional[subprocess.Popen] = None
        logs_thread: Optional[threading.Thread] = None
        try:
            import modal

            app_name = context.modal_app_name or os.environ.get("TS_AGENTS_MODAL_APP") or self.app_name
            fn_name = context.modal_function_name or os.environ.get("TS_AGENTS_MODAL_FUNCTION") or self.function_name
            environment_name = context.modal_environment or os.environ.get("MODAL_ENVIRONMENT") or None
            stream_logs = self._coerce_bool_env("TS_AGENTS_MODAL_STREAM", context.modal_stream_logs)
            modal_timestamps = self._coerce_bool_env("TS_AGENTS_MODAL_LOG_TIMESTAMPS", False)

            self._emit_log(
                f"[modal] invoking app={app_name} function={fn_name} env={environment_name or '(default)'}\n",
                modal_log_handle,
            )
            if stream_logs:
                logs_proc, logs_thread = self._start_modal_log_stream(
                    app_name=app_name,
                    environment_name=environment_name,
                    log_file_handle=modal_log_handle,
                    timestamps=modal_timestamps,
                )

            fn = modal.Function.from_name(app_name, fn_name, environment_name=environment_name)

            request = {
                "tool_name": tool_name,
                "kwargs": params,
                "context": {
                    "timeout_seconds": context.timeout_seconds,
                    "memory_mb": context.memory_mb,
                    "disk_mb": context.disk_mb,
                    "working_dir": context.working_dir,
                },
            }

            response_dict = fn.remote(request)
            self._emit_log("[modal] remote call completed\n", modal_log_handle)
            result = ExecutionResult.from_dict(response_dict)
            result.metadata = {
                **(result.metadata or {}),
                "backend": "modal",
                "app": app_name,
                "function": fn_name,
                "environment": environment_name,
            }
            return result

        except Exception as e:
            self._emit_log(f"[modal] execution error: {e!r}\n", modal_log_handle)
            err = ToolError.from_exception(e, tool_name=tool_name)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=err,
                formatted_output=str(err),
                duration_ms=(time.time() - start_time) * 1000,
                metadata={"tool_name": tool_name, "backend": "modal"},
            )
        finally:
            if logs_proc is not None:
                try:
                    if logs_proc.poll() is None:
                        logs_proc.terminate()
                        try:
                            logs_proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            logs_proc.kill()
                except Exception:
                    pass
            if logs_thread is not None:
                try:
                    logs_thread.join(timeout=1)
                except Exception:
                    pass
            if modal_log_handle is not None:
                try:
                    modal_log_handle.close()
                except Exception:
                    pass


class ToolExecutor:
    """Main tool executor that routes to appropriate backends.

    This is the central abstraction for tool execution. It:
    - Looks up tools in the registry
    - Validates parameters
    - Routes to the appropriate execution backend
    - Handles result formatting

    Parameters
    ----------
    default_backend : SandboxMode
        Default execution backend
    backends : Dict[SandboxMode, ExecutorBackend], optional
        Custom backend implementations

    Examples
    --------
    >>> executor = ToolExecutor()
    >>>
    >>> # Execute with default (local) backend
    >>> result = executor.execute("stl_decompose_with_data", {
    ...     "variable_name": "bx001_real",
    ...     "unique_id": "Re200Rm200",
    ... })
    >>>
    >>> # Execute with Docker sandbox
    >>> result = executor.execute("stl_decompose_with_data", {
    ...     "variable_name": "bx001_real",
    ...     "unique_id": "Re200Rm200",
    ... }, context=ExecutionContext(sandbox_mode=SandboxMode.DOCKER))
    """

    def __init__(
        self,
        default_backend: SandboxMode = SandboxMode.LOCAL,
        backends: Optional[Dict[SandboxMode, ExecutorBackend]] = None,
    ):
        self.default_backend = default_backend
        self.backends: Dict[SandboxMode, ExecutorBackend] = backends or {
            SandboxMode.LOCAL: LocalBackend(),
            SandboxMode.DOCKER: DockerBackend(),
            SandboxMode.DAYTONA: DaytonaBackend(),
            SandboxMode.MODAL: ModalBackend(),
            SandboxMode.SUBPROCESS: SubprocessBackend(),
        }
        self._execution_log: List[Dict[str, Any]] = []

    def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> ExecutionResult:
        """Execute a tool by name.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute
        params : Dict[str, Any]
            Parameters to pass to the tool
        context : ExecutionContext, optional
            Execution context (defaults to local execution)

        Returns
        -------
        ExecutionResult
            The execution result
        """
        from .registry import ToolRegistry

        context = context or ExecutionContext(sandbox_mode=self.default_backend)

        try:
            context.sandbox_mode = _coerce_sandbox_mode(context.sandbox_mode)
        except ValueError as exc:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=ToolError(
                    code=ToolErrorCode.VALIDATION_ERROR,
                    message=str(exc),
                    recoverable=True,
                    tool_name=tool_name,
                ),
            )

        # Look up tool in registry
        try:
            metadata = ToolRegistry.get(tool_name)
        except KeyError as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=ToolError(
                    code=ToolErrorCode.NOT_FOUND,
                    message=str(e),
                    tool_name=tool_name,
                ),
            )

        # Validate parameters
        validation_error = self._validate_params(tool_name, params, metadata)
        if validation_error:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=validation_error,
            )

        # Check if approval is needed for high-cost tools
        if self._needs_approval(metadata, context):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=ToolError(
                    code=ToolErrorCode.PERMISSION_DENIED,
                    message=f"Tool '{tool_name}' requires user approval for execution",
                    recoverable=True,
                    tool_name=tool_name,
                ),
            )

        # Get execution backend
        backend = self.backends.get(context.sandbox_mode)
        if backend is None or not backend.is_available():
            logger.warning(
                f"Backend {context.sandbox_mode} not available, falling back to local"
            )
            backend = self.backends[SandboxMode.LOCAL]

        # Apply resource limits from metadata if not specified in context
        context = self._apply_metadata_limits(metadata, context)

        # Execute
        result = backend.execute(
            tool_name=tool_name,
            func=metadata.core_function,
            params=params,
            context=context,
        )

        # Log execution
        self._log_execution(tool_name, params, context, result)

        return result

    def _validate_params(
        self,
        tool_name: str,
        params: Dict[str, Any],
        metadata: Any,
    ) -> Optional[ToolError]:
        """Validate parameters against tool metadata."""
        required_params = [p.name for p in metadata.parameters if not p.optional]
        missing = [p for p in required_params if p not in params]

        if missing:
            return ToolError(
                code=ToolErrorCode.VALIDATION_ERROR,
                message=f"Missing required parameters: {', '.join(missing)}",
                recoverable=True,
                tool_name=tool_name,
                details={"missing_params": missing},
            )

        # Run custom validation if available
        if hasattr(metadata, 'input_validation_fn') and metadata.input_validation_fn:
            try:
                metadata.input_validation_fn(params)
            except Exception as e:
                return ToolError(
                    code=ToolErrorCode.VALIDATION_ERROR,
                    message=str(e),
                    recoverable=True,
                    tool_name=tool_name,
                )

        return None

    def _needs_approval(self, metadata: Any, context: ExecutionContext) -> bool:
        """Check if tool execution needs user approval."""
        from .registry import ComputationalCost

        if context.user_approved:
            return False

        # Very high cost tools need approval
        if metadata.cost == ComputationalCost.VERY_HIGH:
            return True

        return False

    def _apply_metadata_limits(
        self,
        metadata: Any,
        context: ExecutionContext,
    ) -> ExecutionContext:
        """Apply resource limits from metadata if not specified."""
        # Check for resource specs in metadata (new fields)
        if context.timeout_seconds is None and hasattr(metadata, 'timeout_seconds'):
            context.timeout_seconds = metadata.timeout_seconds
        if context.memory_mb is None and hasattr(metadata, 'memory_mb'):
            context.memory_mb = metadata.memory_mb
        if context.disk_mb is None and hasattr(metadata, 'disk_mb'):
            context.disk_mb = metadata.disk_mb

        return context

    def _log_execution(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: ExecutionContext,
        result: ExecutionResult,
    ) -> None:
        """Log tool execution for analytics."""
        entry = {
            "tool_name": tool_name,
            "params": {k: str(v)[:100] for k, v in params.items()},
            "sandbox_mode": context.sandbox_mode.value,
            "status": result.status.value,
            "duration_ms": result.duration_ms,
            "timestamp": datetime.now().isoformat(),
        }
        if result.error:
            entry["error"] = result.error.to_dict()

        self._execution_log.append(entry)
        logger.debug(f"Tool execution: {entry}")

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution log."""
        return self._execution_log.copy()

    def clear_execution_log(self) -> None:
        """Clear the execution log."""
        self._execution_log.clear()


# Default global executor instance
_default_executor: Optional[ToolExecutor] = None


def get_executor() -> ToolExecutor:
    """Get the default executor instance."""
    global _default_executor
    if _default_executor is None:
        _default_executor = ToolExecutor()
    return _default_executor


def execute_tool(
    tool_name: str,
    params: Dict[str, Any],
    context: Optional[ExecutionContext] = None,
) -> ExecutionResult:
    """Convenience function to execute a tool using the default executor.

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute
    params : Dict[str, Any]
        Parameters to pass to the tool
    context : ExecutionContext, optional
        Execution context

    Returns
    -------
    ExecutionResult
        The execution result
    """
    return get_executor().execute(tool_name, params, context)
