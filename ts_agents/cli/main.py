"""CLI entrypoint and subcommand handlers."""

from __future__ import annotations

import argparse
from difflib import get_close_matches
from functools import partial
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from ts_agents.cli_contracts import normalize_cli_template
from ts_agents.contracts import CLIEnvelope, CLIError, CLIExecution
from ts_agents.tools.executor import ToolError, ToolErrorCode

from .output import (
    dump_json,
    extract_images_from_jsonable,
    extract_images_to_files,
    render_output,
    to_jsonable,
    write_output,
)


_INSTALL_EXTRA_METADATA: Dict[str, Dict[str, Any]] = {
    "viz": {
        "install_spec": "ts-agents[viz]",
        "dependencies": ["matplotlib"],
        "description": "Plot artifacts and visualization helpers.",
    },
    "ui": {
        "install_spec": "ts-agents[ui]",
        "dependencies": ["gradio", "matplotlib"],
        "description": "Gradio UI and hosted/manual demo entrypoints.",
    },
    "agents": {
        "install_spec": "ts-agents[agents]",
        "dependencies": ["langchain", "langchain_core", "langchain_openai"],
        "description": "Built-in agent entrypoints and orchestration adapters.",
    },
    "decomposition": {
        "install_spec": "ts-agents[decomposition]",
        "dependencies": ["statsmodels"],
        "description": "Statsmodels-backed decomposition methods.",
    },
    "forecasting": {
        "install_spec": "ts-agents[forecasting]",
        "dependencies": ["statsforecast"],
        "description": "StatsForecast-backed ARIMA/ETS/Theta forecasting methods.",
    },
    "patterns": {
        "install_spec": "ts-agents[patterns]",
        "dependencies": ["ruptures", "stumpy"],
        "description": "Pattern, changepoint, and matrix-profile tooling.",
    },
    "classification": {
        "install_spec": "ts-agents[classification]",
        "dependencies": ["aeon", "sklearn"],
        "description": "Activity-recognition and classifier evaluation workflows.",
    },
}

_INSTALL_PROFILE_GROUPS: Dict[str, List[str]] = {
    "recommended": [
        "classification",
        "ui",
        "agents",
        "viz",
        "forecasting",
        "decomposition",
    ],
    "all": [
        "classification",
        "ui",
        "agents",
        "viz",
        "forecasting",
        "decomposition",
        "patterns",
    ],
}


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw}")


class TSArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that propagates exit_on_error to nested subparsers."""

    def __init__(self, *args, **kwargs):
        self._ts_exit_on_error = kwargs.get("exit_on_error", True)
        super().__init__(*args, **kwargs)

    def add_subparsers(self, **kwargs):
        kwargs.setdefault(
            "parser_class",
            partial(type(self), exit_on_error=self._ts_exit_on_error),
        )
        return super().add_subparsers(**kwargs)


def _module_is_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def _detect_install_profile() -> Dict[str, Any]:
    extras: Dict[str, Dict[str, Any]] = {}
    active_extras: List[str] = []
    for extra, metadata in _INSTALL_EXTRA_METADATA.items():
        missing_dependencies = [
            dependency
            for dependency in metadata["dependencies"]
            if not _module_is_available(dependency)
        ]
        available = not missing_dependencies
        if available:
            active_extras.append(extra)
        extras[extra] = {
            "install_spec": metadata["install_spec"],
            "description": metadata["description"],
            "available": available,
            "missing_dependencies": missing_dependencies,
        }

    if all(extras[extra]["available"] for extra in _INSTALL_PROFILE_GROUPS["all"]):
        current_profile = "all"
    elif all(extras[extra]["available"] for extra in _INSTALL_PROFILE_GROUPS["recommended"]):
        current_profile = "recommended"
    elif active_extras:
        current_profile = "custom"
    else:
        current_profile = "base"

    return {
        "package": "ts-agents",
        "base_install": "ts-agents",
        "current_profile": current_profile,
        "recommended_install": "ts-agents[recommended]",
        "all_features_install": "ts-agents[all]",
        "active_extras": active_extras,
        "profiles": {
            "base": {
                "install_spec": "ts-agents",
                "description": (
                    "CLI-first base install with discovery, inspect-series, and the "
                    "seasonal_naive forecasting baseline."
                ),
            },
            "recommended": {
                "install_spec": "ts-agents[recommended]",
                "description": "Demo-friendly profile covering the main workflow stack.",
                "extras": list(_INSTALL_PROFILE_GROUPS["recommended"]),
            },
            "all": {
                "install_spec": "ts-agents[all]",
                "description": "Full optional feature set, including patterns tooling.",
                "extras": list(_INSTALL_PROFILE_GROUPS["all"]),
            },
        },
        "extras": extras,
    }


def _maybe_number(raw: str) -> Any:
    try:
        if raw.strip().isdigit():
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def _normalize_param_type(param_type: str) -> str:
    return param_type.replace(" ", "").lower()


def _looks_like_list(raw: str) -> bool:
    stripped = raw.strip()
    return stripped.startswith("[") or ("," in stripped and not stripped.startswith("{"))


def _parse_list_value(raw: str, subtype: Optional[str]) -> Any:
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list for list type, got {type(parsed).__name__}")
        items = parsed
    elif raw == "":
        items = []
    else:
        items = [p.strip() for p in raw.split(",")]

    if subtype in {"int", "integer"}:
        return [int(p) for p in items]
    if subtype in {"float", "number"}:
        return [float(p) for p in items]
    if subtype in {"str", "string"}:
        return [str(p) for p in items]
    return [_maybe_number(str(p)) for p in items]


def _parse_param_value(raw: str, param_type: str) -> Any:
    if raw is None:
        return None

    raw = raw.strip()
    if raw.lower() in {"none", "null"}:
        return None

    kind = _normalize_param_type(param_type)

    if "|" in kind:
        options = [opt for opt in kind.split("|") if opt]
        list_like = _looks_like_list(raw)
        if list_like:
            list_opts = [opt for opt in options if "list" in opt or "array" in opt]
            ordered = list_opts + [opt for opt in options if opt not in list_opts]
        else:
            non_list_opts = [opt for opt in options if "list" not in opt and "array" not in opt]
            ordered = non_list_opts + [opt for opt in options if opt not in non_list_opts]

        last_exc: Optional[Exception] = None
        for opt in ordered:
            try:
                return _parse_param_value(raw, opt)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc
        raise ValueError(f"Invalid parameter value '{raw}' for type {param_type}")

    if kind in {"bool", "boolean"}:
        return _parse_bool(raw)

    if kind in {"int", "integer"}:
        return int(raw)

    if kind in {"float", "number"}:
        return float(raw)

    if kind in {"dict", "object"}:
        return json.loads(raw)

    if kind.startswith("list"):
        subtype = None
        if kind.startswith("list[") and kind.endswith("]"):
            subtype = kind[5:-1]
        return _parse_list_value(raw, subtype)

    if kind in {"array", "np.ndarray", "numpy.ndarray", "ndarray"}:
        return _parse_list_value(raw, None)

    return raw


def _parse_param_entries(
    entries: List[str],
    param_types: Dict[str, str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --param entry '{entry}', expected key=value")
        key, raw = entry.split("=", 1)
        key = key.strip()
        if key not in param_types:
            raise ValueError(
                f"Unknown parameter '{key}'. Available: {', '.join(param_types.keys())}"
            )
        params[key] = _parse_param_value(raw, param_types[key])
    return params


def _apply_run_var_args(
    params: Dict[str, Any],
    param_types: Dict[str, str],
    run_value: Optional[str],
    var_value: Optional[str],
) -> Dict[str, Any]:
    if run_value:
        if "unique_id" in param_types and "unique_id" not in params:
            params["unique_id"] = run_value
        if "run_id" in param_types and "run_id" not in params:
            params["run_id"] = run_value

    if var_value:
        if "variable_name" in param_types and "variable_name" not in params:
            params["variable_name"] = var_value
        if "variable" in param_types and "variable" not in params:
            params["variable"] = var_value

    return params


def _suggest_tool_names(tool_name: str, candidate_names: List[str], limit: int = 3) -> List[str]:
    """Suggest nearby tool names for typo recovery."""
    if not tool_name:
        return []
    return get_close_matches(tool_name, candidate_names, n=limit, cutoff=0.45)


def _build_param_placeholder(name: str, param_type: str) -> str:
    kind = _normalize_param_type(param_type)
    if "int" in kind:
        return "1"
    if "float" in kind or "number" in kind:
        return "0.1"
    if "bool" in kind:
        return "true"
    if "list" in kind or "array" in kind:
        return "a,b"
    if "dict" in kind or "object" in kind:
        return '{"key":"value"}'
    return "value"


def _build_run_example_command(
    tool_name: str,
    required: List[str],
    param_types: Dict[str, str],
) -> str:
    parts = [f"uv run ts-agents tool run {tool_name}"]

    # Prefer CLI shorthands where available.
    if "unique_id" in required:
        parts.append("--run Re200Rm200")
    if "run_id" in required and "unique_id" not in required:
        parts.append("--run Re200Rm200")
    if "variable_name" in required:
        parts.append("--var bx001_real")
    if "variable" in required and "variable_name" not in required:
        parts.append("--var bx001_real")

    for name in required:
        if name in {"unique_id", "run_id", "variable_name", "variable"}:
            continue
        placeholder = _build_param_placeholder(name, param_types.get(name, "str"))
        param_value = f"{name}={placeholder}"
        parts.append(f"--param {shlex.quote(param_value)}")

    return " ".join(parts)


def _raise_missing_required_error(
    *,
    tool_name: str,
    param_types: Dict[str, str],
    required: List[str],
    provided: Dict[str, Any],
) -> None:
    missing = [name for name in required if name not in provided]
    if not missing:
        return

    missing_text = ", ".join(missing)
    required_text = ", ".join(required) if required else "(none)"
    example = _build_run_example_command(tool_name, required, param_types)
    raise ValueError(
        f"Missing required parameters for '{tool_name}': {missing_text}\n"
        f"Required parameters: {required_text}\n"
        f"Example:\n  {example}\n"
        f"Tip: inspect tool parameters with:\n  uv run ts-agents tool list --json"
    )


def _raise_unknown_tool_error(tool_name: str) -> NoReturn:
    from ts_agents.tools.registry import ToolRegistry

    tool_names = [tool.name for tool in ToolRegistry.list_all()]
    suggestions = _suggest_tool_names(tool_name, tool_names)
    if suggestions:
        hint = f" Did you mean: {', '.join(suggestions)}?"
    else:
        hint = ""
    raise ValueError(
        f"Tool '{tool_name}' not found.{hint}\n"
        f"Tip: list tools with:\n"
        f"  uv run ts-agents tool list\n"
        f"  uv run ts-agents tool list --json"
    )


def _raise_unknown_workflow_error(workflow_name: str) -> NoReturn:
    from ts_agents.workflows import list_workflows

    workflow_names = [workflow.name for workflow in list_workflows()]
    suggestions = _suggest_tool_names(workflow_name, workflow_names)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ValueError(
        f"Workflow '{workflow_name}' not found.{hint}\n"
        f"Tip: inspect workflows with:\n"
        f"  uv run ts-agents workflow list"
    )


def _add_json_input_args(parser: argparse.ArgumentParser) -> None:
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="JSON object/string or path to JSON file used as structured input",
    )
    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read structured JSON input from stdin",
    )


def _add_sandbox_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sandbox",
        choices=["local", "subprocess", "docker", "daytona", "modal"],
        default=None,
        help="Execution sandbox (overrides TS_AGENTS_SANDBOX_MODE)",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network access in sandboxed execution (if supported)",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow fallback to another backend when the requested sandbox is unavailable",
    )
    parser.add_argument(
        "--fallback-backend",
        choices=["local", "subprocess", "docker", "daytona", "modal"],
        default=None,
        help="Fallback backend to use with --allow-fallback (default: local)",
    )


def _add_workflow_run_lifecycle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reusing an existing explicit output directory by clearing prior artifacts first.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume into an existing explicit output directory that already has a workflow manifest.",
    )


def _add_tool_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tool", type=str, help="Tool name to execute")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Tool parameter in key=value form (repeatable)",
    )
    parser.add_argument(
        "--run",
        dest="run_id",
        type=str,
        default=None,
        help="Run ID (maps to unique_id/run_id)",
    )
    parser.add_argument(
        "--var",
        dest="variable",
        type=str,
        default=None,
        help="Variable name (maps to variable_name/variable)",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve execution of very high cost tools",
    )
    _add_sandbox_execution_args(parser)
    _add_json_input_args(parser)
    _add_output_args(parser)


def _command_label(args: argparse.Namespace) -> str:
    if args.command == "capabilities":
        return "capabilities"
    if args.command == "tool":
        return f"tool {args.tool_command}"
    if args.command == "skills":
        return f"skills {args.skills_command}"
    if args.command == "sandbox":
        return f"sandbox {args.sandbox_command}"
    if args.command == "workflow":
        return f"workflow {args.workflow_command}"
    if args.command == "run":
        return "tool run"
    return args.command


def _command_target_name(args: argparse.Namespace, exc: Optional[Exception] = None) -> Optional[str]:
    if args.command == "tool":
        if args.tool_command in {"show", "run"}:
            return getattr(args, "tool", None)
        if args.tool_command == "search":
            return getattr(args, "query", None)
    if args.command == "sandbox":
        if args.sandbox_command == "doctor":
            return getattr(args, "backend", None)
    if args.command == "workflow":
        if args.workflow_command in {"run", "show"}:
            return getattr(args, "workflow_name", None)
    if args.command == "skills":
        if args.skills_command == "show":
            return getattr(args, "skill", None)
    if args.command == "run":
        return getattr(args, "tool", None)
    if isinstance(exc, ToolError):
        return exc.tool_name
    return None


def _command_input_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if hasattr(args, "_ts_input_payload"):
        return getattr(args, "_ts_input_payload")

    if args.command == "capabilities":
        return {}
    if args.command == "tool":
        if args.tool_command == "list":
            return {
                "bundle": args.bundle,
                "category": args.category,
                "max_cost": args.max_cost,
            }
        if args.tool_command == "show":
            return {"tool": args.tool}
        if args.tool_command == "search":
            return {
                "query": args.query,
                "category": args.category,
                "max_cost": args.max_cost,
            }
    if args.command == "data":
        return {k: v for k, v in vars(args).items() if k not in {"json", "save", "extract_images"}}
    if args.command == "sandbox":
        return {
            k: v
            for k, v in vars(args).items()
            if k not in {"json", "save", "extract_images"}
        }
    if args.command == "skills":
        return {k: v for k, v in vars(args).items() if k not in {"json", "save", "extract_images"}}
    if args.command == "workflow":
        return {
            k: v
            for k, v in vars(args).items()
            if k
            not in {
                "json",
                "save",
                "extract_images",
                "_ts_input_payload",
                "_ts_execution_result",
                "_ts_raw_argv",
            }
        }
    if args.command == "demo":
        return {k: v for k, v in vars(args).items() if k not in {"json", "save", "extract_images"}}
    if args.command == "agent":
        return {
            "type": args.type,
            "model": args.model,
            "tool_bundle": args.tool_bundle,
        }
    if args.command == "run":
        return getattr(args, "_ts_input_payload", {})
    return {}


def _execution_payload(execution: Any) -> Optional[CLIExecution]:
    if execution is None:
        return None

    metadata = getattr(execution, "metadata", {}) or {}
    backend_requested = metadata.get("backend_requested")
    backend_actual = metadata.get("backend_actual") or metadata.get("backend")
    return CLIExecution(
        backend_requested=backend_requested,
        backend_actual=backend_actual,
        duration_ms=getattr(execution, "duration_ms", None),
        metadata=metadata,
    )


def _workflow_execution_metadata(execution: Any) -> Dict[str, Any]:
    metadata = getattr(execution, "metadata", {}) or {}
    backend_requested = metadata.get("backend_requested")
    backend_actual = metadata.get("backend_actual") or metadata.get("backend")
    execution_metadata = {
        "backend_requested": backend_requested,
        "backend_actual": backend_actual,
        "fallback_allowed": metadata.get("fallback_allowed"),
        "fallback_used": metadata.get("fallback_used"),
        "fallback_backend": metadata.get("fallback_backend"),
    }
    return {key: value for key, value in execution_metadata.items() if value is not None}


def _synchronize_workflow_manifest(result: Any, execution: Any) -> None:
    if isinstance(result, dict):
        data = result.get("data")
        status = result.get("status")
        summary = result.get("summary")
        warnings = result.get("warnings") or []
        artifacts = result.get("artifacts")
        provenance = result.get("provenance")
    else:
        data = getattr(result, "data", None)
        status = getattr(result, "status", None)
        summary = getattr(result, "summary", None)
        warnings = getattr(result, "warnings", []) or []
        artifacts = getattr(result, "artifacts", None)
        provenance = getattr(result, "provenance", None)

    if not isinstance(data, dict):
        return

    execution_metadata = _workflow_execution_metadata(execution)
    if execution_metadata:
        data["execution"] = dict(execution_metadata)
        run_metadata = data.get("run")
        if isinstance(run_metadata, dict):
            run_metadata["execution"] = dict(execution_metadata)

    manifest_path_raw = data.get("manifest_path")
    if not isinstance(manifest_path_raw, str):
        return

    manifest_path = Path(manifest_path_raw)
    if not manifest_path.exists():
        return

    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(manifest_payload, dict):
        return

    if isinstance(status, str):
        manifest_payload["status"] = status
    if isinstance(summary, str):
        manifest_payload["summary"] = summary
    if isinstance(data.get("output_dir"), str):
        manifest_payload["output_dir"] = data["output_dir"]
    manifest_payload["manifest_path"] = str(manifest_path)
    if isinstance(data.get("run_id"), str):
        manifest_payload["run_id"] = data["run_id"]
    if isinstance(data.get("source"), dict):
        manifest_payload["source"] = to_jsonable(data["source"])
    manifest_payload["warnings"] = to_jsonable(warnings)
    manifest_payload["quality_flags"] = to_jsonable(data.get("quality_flags") or [])
    if isinstance(artifacts, list):
        manifest_payload["artifacts"] = to_jsonable(artifacts)
    if provenance is not None:
        manifest_payload["provenance"] = to_jsonable(provenance)
    if execution_metadata:
        manifest_payload["execution"] = dict(execution_metadata)

    payload_text = render_output(to_jsonable(manifest_payload), json_output=True)
    write_output(payload_text, str(manifest_path))


def _quality_payload(result: Any) -> Tuple[Optional[str], Optional[bool], Optional[bool]]:
    if isinstance(result, dict):
        status = result.get("status")
        warnings = result.get("warnings") or []
        data = result.get("data") or {}
    else:
        status = getattr(result, "status", None)
        warnings = getattr(result, "warnings", []) or []
        data = getattr(result, "data", {}) or {}

    if isinstance(data, dict):
        quality_flags = data.get("quality_flags") or []
    else:
        quality_flags = getattr(data, "quality_flags", []) or []
    if status is None and not warnings and not quality_flags:
        return None, None, None

    degraded = status == "degraded"
    requires_review = degraded or bool(warnings) or bool(quality_flags)
    if status in {"ok", "degraded"}:
        quality_status = status
    elif requires_review:
        quality_status = "review"
    else:
        quality_status = None
    return quality_status, degraded, requires_review


def _success_envelope(args: argparse.Namespace, result: Any) -> CLIEnvelope:
    execution = _execution_payload(getattr(args, "_ts_execution_result", None))
    quality_status, degraded, requires_review = _quality_payload(result)
    return CLIEnvelope(
        ok=True,
        command=_command_label(args),
        name=_command_target_name(args),
        input=_command_input_payload(args),
        quality_status=quality_status,
        degraded=degraded,
        requires_review=requires_review,
        result=result,
        execution=execution,
    )


def _exception_to_cli_error(exc: Exception) -> CLIError:
    if isinstance(exc, argparse.ArgumentError):
        argument = getattr(exc, "argument_name", None)
        details = {"usage_error": True}
        if argument:
            details["argument"] = argument
        return CLIError(
            code="usage_error",
            message=str(exc),
            retryable=False,
            details=details,
        )

    if isinstance(exc, ToolError):
        return CLIError(
            code=exc.code.value,
            message=exc.message,
            retryable=exc.recoverable,
            hint=exc.hint,
            details=exc.details,
        )

    if isinstance(exc, ValueError):
        return CLIError(code="validation_error", message=str(exc), retryable=False)

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return CLIError(code="dependency_error", message=str(exc), retryable=False)

    if isinstance(exc, PermissionError):
        return CLIError(code="permission_denied", message=str(exc), retryable=False)

    if isinstance(exc, TimeoutError):
        return CLIError(code="timeout", message=str(exc), retryable=True)

    return CLIError(code="execution_failure", message=str(exc), retryable=False)


def _error_envelope(args: argparse.Namespace, exc: Exception) -> CLIEnvelope:
    execution = _execution_payload(getattr(args, "_ts_execution_result", None))
    return CLIEnvelope(
        ok=False,
        command=_command_label(args),
        name=_command_target_name(args, exc),
        input=_command_input_payload(args),
        error=_exception_to_cli_error(exc),
        execution=execution,
    )


def _exit_code_for_exception(exc: Exception) -> int:
    if isinstance(exc, argparse.ArgumentError):
        return 2

    if isinstance(exc, ToolError):
        mapping = {
            ToolErrorCode.VALIDATION_ERROR: 2,
            ToolErrorCode.DEPENDENCY_ERROR: 3,
            ToolErrorCode.DATA_ERROR: 4,
            ToolErrorCode.NOT_FOUND: 4,
            ToolErrorCode.BACKEND_UNAVAILABLE: 5,
            ToolErrorCode.PERMISSION_DENIED: 7,
            ToolErrorCode.TIMEOUT: 8,
            ToolErrorCode.RESOURCE_EXHAUSTED: 8,
        }
        return mapping.get(exc.code, 6)

    if isinstance(exc, ValueError):
        return 2
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return 3
    if isinstance(exc, PermissionError):
        return 7
    if isinstance(exc, TimeoutError):
        return 8
    return 6


def _argv_requests_json(argv: Optional[List[str]]) -> bool:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    return "--json" in raw_argv


def _flag_was_provided(args: argparse.Namespace, flag: str) -> bool:
    raw_argv = getattr(args, "_ts_raw_argv", []) or []
    return any(token == flag or token.startswith(f"{flag}=") for token in raw_argv)


def _workflow_default_output_root(workflow: Any) -> str:
    for option in getattr(workflow, "options", []):
        if option.name == "output_dir" and option.default:
            return str(option.default)
    return f"outputs/{getattr(workflow, 'name', 'workflow')}"


def _prepare_workflow_run_output(args: argparse.Namespace, workflow: Any) -> Dict[str, Any]:
    from ts_agents.workflows.common import (
        WORKFLOW_MANIFEST_FILENAME,
        generate_workflow_run_id,
        output_dir_has_files,
        read_workflow_manifest,
    )

    overwrite = getattr(args, "overwrite", False)
    resume = getattr(args, "resume", False)
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    explicit_output_dir = _flag_was_provided(args, "--output-dir")
    if (overwrite or resume) and not explicit_output_dir:
        raise ValueError("--overwrite and --resume require an explicit --output-dir.")

    requested_output_dir = getattr(args, "output_dir", None)
    output_dir_mode = "explicit" if explicit_output_dir else "generated"
    if explicit_output_dir:
        output_path = Path(requested_output_dir).expanduser().resolve()
        if output_path.exists() and not output_path.is_dir():
            raise ValueError("--output-dir must point to a directory path.")
    else:
        base_output_root = Path(_workflow_default_output_root(workflow)).expanduser()
        run_id = generate_workflow_run_id()
        output_path = (base_output_root / run_id).resolve()

    existing_manifest = read_workflow_manifest(output_path) if explicit_output_dir and resume else None
    resumed = False
    if explicit_output_dir:
        if resume:
            if not output_path.exists():
                raise ValueError("--resume requires an existing --output-dir.")
            if existing_manifest is None:
                raise ValueError(
                    f"--resume requires {WORKFLOW_MANIFEST_FILENAME} in the output directory."
                )
            if not isinstance(existing_manifest, dict):
                raise ValueError(
                    f"{WORKFLOW_MANIFEST_FILENAME} in {output_path} must contain a JSON object for --resume."
                )
            run_id = str(existing_manifest.get("run_id") or generate_workflow_run_id())
            resumed = True
        else:
            run_id = generate_workflow_run_id()
            if output_dir_has_files(output_path) and not overwrite:
                raise ValueError(
                    "Output directory already exists and is not empty. "
                    "Use --overwrite to replace prior artifacts or --resume to continue an existing run."
                )
    else:
        if output_path.exists():
            raise ValueError(
                "Generated workflow output directory already exists unexpectedly; retry the command."
            )

    args.output_dir = str(output_path)
    return {
        "run_id": run_id,
        "resumed": resumed,
        "output_dir_mode": output_dir_mode,
        "manifest_path": str(output_path / WORKFLOW_MANIFEST_FILENAME),
        "output_dir": str(output_path),
        "_clear_output_dir": bool(explicit_output_dir and overwrite and output_path.exists()),
    }


def _materialize_workflow_output_dir(run_lifecycle: Dict[str, Any]) -> None:
    from ts_agents.workflows.common import clear_output_dir

    if run_lifecycle.get("_clear_output_dir"):
        clear_output_dir(run_lifecycle["output_dir"])


def _infer_command_label_from_argv(argv: List[str]) -> str:
    if not argv:
        return "ts-agents"

    command = argv[0]
    if command in {"tool", "workflow", "data", "sandbox", "skills", "agent", "demo", "capabilities"}:
        if len(argv) > 1 and not argv[1].startswith("-"):
            return f"{command} {argv[1]}"
    return command


def _infer_target_name_from_argv(argv: List[str]) -> Optional[str]:
    if not argv:
        return None

    command = argv[0]
    if command == "tool":
        if len(argv) > 2 and argv[1] in {"run", "show"} and not argv[2].startswith("-"):
            return argv[2]
        if len(argv) > 2 and argv[1] == "search" and not argv[2].startswith("-"):
            return argv[2]
        return None

    if command == "workflow":
        if len(argv) > 2 and argv[1] in {"run", "show"} and not argv[2].startswith("-"):
            return argv[2]
        return None

    if command == "skills":
        if len(argv) > 2 and argv[1] == "show" and not argv[2].startswith("-"):
            return argv[2]
        return None

    if command in {"run", "demo"} and len(argv) > 1 and not argv[1].startswith("-"):
        return argv[1]

    return None


def _parse_error_envelope(argv: List[str], exc: argparse.ArgumentError) -> CLIEnvelope:
    return CLIEnvelope(
        ok=False,
        command=_infer_command_label_from_argv(argv),
        name=_infer_target_name_from_argv(argv),
        input={"argv": argv},
        error=_exception_to_cli_error(exc),
        execution=None,
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable text",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save output",
    )
    parser.add_argument(
        "--extract-images",
        type=str,
        default=None,
        help="Extract embedded [IMAGE_DATA:...] payloads into PNG files under this directory (requires --save)",
    )


def _add_data_subcommands(subparsers: argparse._SubParsersAction) -> None:
    data_parser = subparsers.add_parser("data", help="Bundled data utilities")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    list_parser = data_sub.add_parser("list", help="List runs and variables")
    list_parser.add_argument(
        "--data-type",
        choices=["real", "imag"],
        default="real",
        help="Dataset type (default: real)",
    )
    list_parser.add_argument(
        "--runs",
        action="store_true",
        help="Only list run IDs",
    )
    list_parser.add_argument(
        "--variables",
        action="store_true",
        help="Only list variables",
    )
    data_group = list_parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--use-test-data",
        action="store_true",
        help="Force use of test data",
    )
    data_group.add_argument(
        "--full-data",
        action="store_true",
        help="Force use of full dataset",
    )
    _add_output_args(list_parser)

    vars_parser = data_sub.add_parser("vars", help="List variables (alias for list --variables)")
    vars_parser.add_argument(
        "--data-type",
        choices=["real", "imag"],
        default="real",
        help="Dataset type (default: real)",
    )
    data_group = vars_parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--use-test-data",
        action="store_true",
        help="Force use of test data",
    )
    data_group.add_argument(
        "--full-data",
        action="store_true",
        help="Force use of full dataset",
    )
    _add_output_args(vars_parser)


def _add_capabilities_subcommand(subparsers: argparse._SubParsersAction) -> None:
    capabilities_parser = subparsers.add_parser(
        "capabilities",
        help="Show machine-readable CLI capabilities for autonomous agents",
    )
    _add_output_args(capabilities_parser)


def _add_tool_subcommands(subparsers: argparse._SubParsersAction) -> None:
    tool_parser = subparsers.add_parser("tool", help="Tool registry commands")
    tool_sub = tool_parser.add_subparsers(dest="tool_command", required=True)

    list_parser = tool_sub.add_parser("list", help="List registered tools")
    list_parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Filter to a named bundle (minimal, standard, full, all, etc.)",
    )
    list_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to a tool category",
    )
    list_parser.add_argument(
        "--max-cost",
        type=str,
        default=None,
        help="Filter to tools at or below a cost (low, medium, high, very_high)",
    )
    _add_output_args(list_parser)

    show_parser = tool_sub.add_parser("show", help="Show detailed metadata for one tool")
    show_parser.add_argument("tool", type=str, help="Tool name to inspect")
    _add_output_args(show_parser)

    search_parser = tool_sub.add_parser("search", help="Search tools by name or description")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional tool category filter",
    )
    search_parser.add_argument(
        "--max-cost",
        type=str,
        default=None,
        help="Optional maximum cost filter",
    )
    _add_output_args(search_parser)

    run_parser = tool_sub.add_parser(
        "run",
        help="Run a tool by name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run ts-agents tool run forecast_theta_with_data --run Re200Rm200 --var bx001_real --param horizon=12\n"
            "  uv run ts-agents tool run describe_series --input-json '{\"series\": [1,2,3,4]}'\n"
            "  echo '{\"series\": [1,2,3,4]}' | uv run ts-agents tool run describe_series --stdin\n"
            "  uv run ts-agents tool run compare_forecasts_with_data --run Re200Rm200 --var bx001_real --param models=arima,theta --json\n"
            "  uv run ts-agents tool run describe_series_with_data --run Re200Rm200 --var bx001_real"
        ),
    )
    _add_tool_run_args(run_parser)


def _add_run_subcommands(subparsers: argparse._SubParsersAction) -> None:
    run_parser = subparsers.add_parser(
        "run",
        help="Deprecated compatibility alias for 'tool run'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Deprecated compatibility alias for 'ts-agents tool run'. Prefer 'ts-agents tool run' in new scripts.\n\n"
            "Examples:\n"
            "  uv run ts-agents tool run forecast_theta_with_data --run Re200Rm200 --var bx001_real --param horizon=12\n"
            "  uv run ts-agents tool run describe_series --input-json '{\"series\": [1,2,3,4]}'\n"
            "  uv run ts-agents tool run compare_forecasts_with_data --run Re200Rm200 --var bx001_real --param models=arima,theta --json\n"
            "  uv run ts-agents tool run stl_decompose_with_data --run Re200Rm200 --var bx001_real --save outputs/stl.md"
        ),
    )
    _add_tool_run_args(run_parser)


def _add_agent_subcommands(subparsers: argparse._SubParsersAction) -> None:
    agent_parser = subparsers.add_parser("agent", help="Agent workflows")
    agent_sub = agent_parser.add_subparsers(dest="agent_command", required=True)

    run_parser = agent_sub.add_parser(
        "run",
        help="Run a single agent prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run ts-agents agent run \"Find peaks in bx001_real for Re200Rm200\"\n"
            "  uv run ts-agents agent run --type simple --tool-bundle demo \"Run the windowing demo workflow\"\n"
            "  uv run ts-agents agent run --type deep --approval auto \"Compare forecasting methods for bx001_real\""
        ),
    )
    run_parser.add_argument("prompt", type=str, help="Prompt to send")
    run_parser.add_argument(
        "--type",
        choices=["simple", "deep"],
        default="simple",
        help="Agent type (default: simple)",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name",
    )
    run_parser.add_argument(
        "--tool-bundle",
        type=str,
        default="standard",
        help="Tool bundle for simple agent",
    )
    run_parser.add_argument(
        "--approval",
        choices=["off", "auto", "prompt"],
        default="off",
        help="Deep agent approval handling (default: off)",
    )
    _add_output_args(run_parser)


def _add_skills_subcommands(subparsers: argparse._SubParsersAction) -> None:
    skills_parser = subparsers.add_parser("skills", help="Skills export and validation")
    skills_sub = skills_parser.add_subparsers(dest="skills_command", required=True)

    # Export subcommand
    export_parser = skills_sub.add_parser("export", help="Export skills")
    export_parser.add_argument(
        "--out",
        type=str,
        default="skills_export",
        help="Output directory or filename (default: skills_export)",
    )
    export_parser.add_argument(
        "--agent",
        type=str,
        choices=["claude", "codex", "gemini", "windsurf", "github"],
        default=None,
        help="Export to specific agent's skill directory",
    )
    export_parser.add_argument(
        "--all-agents",
        action="store_true",
        help="Export to all agent-specific directories",
    )
    export_parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copies for agent exports (local dev only)",
    )
    export_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default=None,
        help="Export aggregate skills as markdown or machine-readable JSON (default: infer from output path)",
    )
    _add_output_args(export_parser)

    # Validate subcommand
    validate_parser = skills_sub.add_parser("validate", help="Validate skill files")
    validate_parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Skills directory to validate (default: canonical skills/)",
    )
    _add_output_args(validate_parser)

    # List subcommand
    list_parser = skills_sub.add_parser("list", help="List available skills")
    list_parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Skills directory to list (default: canonical skills/)",
    )
    _add_output_args(list_parser)

    show_parser = skills_sub.add_parser("show", help="Show one skill as structured metadata")
    show_parser.add_argument("skill", type=str, help="Skill name to inspect")
    show_parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Skills directory to inspect (default: canonical skills/)",
    )
    _add_output_args(show_parser)


def _add_sandbox_subcommands(subparsers: argparse._SubParsersAction) -> None:
    sandbox_parser = subparsers.add_parser(
        "sandbox",
        help="Inspect sandbox backend readiness and policy",
    )
    sandbox_sub = sandbox_parser.add_subparsers(dest="sandbox_command", required=True)

    list_parser = sandbox_sub.add_parser("list", help="List sandbox backends and readiness")
    _add_output_args(list_parser)

    doctor_parser = sandbox_sub.add_parser("doctor", help="Probe one sandbox backend")
    doctor_parser.add_argument(
        "backend",
        choices=["local", "subprocess", "docker", "daytona", "modal"],
        help="Backend name to inspect",
    )
    _add_output_args(doctor_parser)


def _add_workflow_source_args(parser: argparse.ArgumentParser) -> None:
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CSV, parquet, or JSON input data. Mutually exclusive with --input-json/--stdin/--run-id.",
    )
    source_group.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="JSON object/string or path to JSON file used as workflow input. Mutually exclusive with --input/--stdin/--run-id.",
    )
    source_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read workflow input from stdin. Mutually exclusive with --input/--input-json/--run-id.",
    )
    source_group.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Bundled dataset run ID. Requires --variable and is mutually exclusive with --input/--input-json/--stdin.",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=None,
        help="Bundled dataset variable name. Requires --run-id and should not be combined with other input source flags.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=None,
        help="Optional time/index column for tabular inputs",
    )
    parser.add_argument(
        "--value-col",
        type=str,
        default=None,
        help="Series value column for tabular inputs",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--use-test-data",
        action="store_true",
        help="Force use of bundled test data when using --run-id/--variable",
    )
    data_group.add_argument(
        "--full-data",
        action="store_true",
        help="Force use of the full bundled dataset when using --run-id/--variable",
    )


def _add_tabular_workflow_source_args(parser: argparse.ArgumentParser) -> None:
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CSV, parquet, JSON, or JSONL input data.",
    )
    source_group.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Tabular JSON payload or path to a JSON file.",
    )
    source_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read tabular JSON or CSV input from stdin.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=None,
        help="Optional time/index column for tabular inputs",
    )


def _add_workflow_subcommands(subparsers: argparse._SubParsersAction) -> None:
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Opinionated end-to-end workflows",
    )
    workflow_sub = workflow_parser.add_subparsers(dest="workflow_command", required=True)

    list_parser = workflow_sub.add_parser("list", help="List available workflows")
    _add_output_args(list_parser)

    show_parser = workflow_sub.add_parser("show", help="Show structured metadata for one workflow")
    show_parser.add_argument("workflow_name", type=str, help="Workflow name to inspect")
    _add_output_args(show_parser)

    run_parser = workflow_sub.add_parser(
        "run",
        help="Run a named workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ts-agents workflow list\n"
            "  ts-agents workflow show forecast-series --json\n"
            "  ts-agents workflow run inspect-series --input data.csv --time-col ds --value-col y\n"
            "  ts-agents workflow run forecast-series --input data.csv --time-col ds --value-col y --horizon 48 --sandbox subprocess\n"
            "  ts-agents workflow run activity-recognition --input stream.csv --label-col label --value-cols x,y,z\n"
            "  echo '{\"series\": [1,2,3,4]}' | ts-agents workflow run inspect-series --stdin"
        ),
    )
    workflow_run_sub = run_parser.add_subparsers(dest="workflow_name", required=True)

    inspect_parser = workflow_run_sub.add_parser(
        "inspect-series",
        help="Run quick diagnostics and write summary/report artifacts",
    )
    _add_workflow_source_args(inspect_parser)
    inspect_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inspect",
        help="Explicit output directory for workflow artifacts. If omitted, a run-scoped subdirectory is created under outputs/inspect.",
    )
    inspect_parser.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Maximum lag for autocorrelation output",
    )
    inspect_parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation",
    )
    _add_workflow_run_lifecycle_args(inspect_parser)
    _add_sandbox_execution_args(inspect_parser)
    _add_output_args(inspect_parser)

    forecast_parser = workflow_run_sub.add_parser(
        "forecast-series",
        help="Compare baseline forecasting methods and write forecast artifacts",
    )
    _add_workflow_source_args(forecast_parser)
    forecast_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/forecast",
        help="Explicit output directory for workflow artifacts. If omitted, a run-scoped subdirectory is created under outputs/forecast.",
    )
    forecast_parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forecast horizon",
    )
    forecast_parser.add_argument(
        "--season-length",
        type=int,
        default=None,
        help="Optional seasonal period for seasonal methods",
    )
    forecast_parser.add_argument(
        "--methods",
        type=str,
        default="seasonal_naive,arima,theta",
        help="Comma-separated methods to compare (default: seasonal_naive,arima,theta)",
    )
    forecast_parser.add_argument(
        "--validation-size",
        type=int,
        default=None,
        help="Holdout size for comparison (default: horizon)",
    )
    forecast_parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation",
    )
    _add_workflow_run_lifecycle_args(forecast_parser)
    _add_sandbox_execution_args(forecast_parser)
    _add_output_args(forecast_parser)

    activity_parser = workflow_run_sub.add_parser(
        "activity-recognition",
        help="Select a window size and evaluate a classifier on a labeled stream",
    )
    _add_tabular_workflow_source_args(activity_parser)
    activity_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/activity",
        help="Explicit output directory for workflow artifacts. If omitted, a run-scoped subdirectory is created under outputs/activity.",
    )
    activity_parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Label column containing per-timepoint activity labels",
    )
    activity_parser.add_argument(
        "--value-cols",
        type=_split_csv,
        default=["x", "y", "z"],
        help="Comma-separated value columns (default: x,y,z)",
    )
    activity_parser.add_argument(
        "--window-sizes",
        type=_split_int_csv,
        default=[32, 64, 96, 128, 160],
        help="Comma-separated candidate window sizes",
    )
    activity_parser.add_argument(
        "--metric",
        type=str,
        default="balanced_accuracy",
        help="Metric: accuracy | balanced_accuracy | f1_macro",
    )
    activity_parser.add_argument(
        "--classifier",
        type=str,
        default="auto",
        help="Classifier: auto | minirocket | rocket | knn",
    )
    activity_parser.add_argument(
        "--balance",
        type=str,
        default="segment_cap",
        help="Balancing: none | undersample | segment_cap",
    )
    activity_parser.add_argument(
        "--max-windows-per-segment",
        type=int,
        default=25,
        help="Cap windows per segment when balance=segment_cap",
    )
    activity_parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test fraction per split",
    )
    activity_parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for selection/evaluation",
    )
    activity_parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation",
    )
    _add_workflow_run_lifecycle_args(activity_parser)
    _add_sandbox_execution_args(activity_parser)
    _add_output_args(activity_parser)


def _resolve_runtime_path(path: str) -> Path:
    from ts_agents.runtime_paths import resolve_existing_path

    resolved = resolve_existing_path(path)
    return resolved if resolved is not None else Path(path)


def _default_demo_csv_path() -> str:
    return str(_resolve_runtime_path("data/demo_labeled_stream.csv"))


def _resolve_required_runtime_path(path: str) -> Path:
    resolved = _resolve_runtime_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required runtime resource not found: {path}")
    return resolved


def _add_demo_subcommands(subparsers: argparse._SubParsersAction) -> None:
    demo_parser = subparsers.add_parser(
        "demo",
        help="Deprecated compatibility demos layered over the workflow surface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Deprecated compatibility surface. Prefer 'ts-agents workflow run ...' for new automation.\n\n"
            "Examples:\n"
            "  uv run ts-agents demo window-classification --no-llm\n"
            "  uv run ts-agents demo forecasting --no-llm\n"
            "  uv run ts-agents demo forecasting --run-id Re200Rm200 --variable bx001_real --horizon 1 --no-llm"
        ),
    )
    demo_sub = demo_parser.add_subparsers(dest="demo_command", required=True)

    window_parser = demo_sub.add_parser(
        "window-classification",
        help="Run the window-size selection + evaluation demo",
    )
    window_parser.add_argument(
        "--csv-path",
        type=str,
        default=_default_demo_csv_path(),
        help="Path to labeled-stream CSV (defaults to bundled demo dataset)",
    )
    window_parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip synthetic data generation step",
    )
    window_parser.add_argument(
        "--scenario",
        type=str,
        default="stairs",
        help="Synthetic data scenario (gait | stairs | industrial, default: stairs)",
    )
    window_parser.add_argument(
        "--hz",
        type=int,
        default=20,
        help="Synthetic data sampling rate (Hz)",
    )
    window_parser.add_argument(
        "--minutes",
        type=float,
        default=4,
        help="Synthetic data duration in minutes",
    )
    window_parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for generation and evaluation",
    )
    window_parser.add_argument(
        "--value-columns",
        type=str,
        default="x,y,z",
        help="Comma-separated value columns (default: x,y,z)",
    )
    window_parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Label column name (default: label)",
    )
    window_parser.add_argument(
        "--window-sizes",
        type=str,
        default="32,64,96,128,160",
        help="Comma-separated candidate window sizes",
    )
    window_parser.add_argument(
        "--metric",
        type=str,
        default="balanced_accuracy",
        help="Metric: accuracy | balanced_accuracy | f1_macro",
    )
    window_parser.add_argument(
        "--classifier",
        type=str,
        default="auto",
        help="Classifier: auto | minirocket | rocket | knn",
    )
    window_parser.add_argument(
        "--balance",
        type=str,
        default="segment_cap",
        help="Balancing: none | undersample | segment_cap",
    )
    window_parser.add_argument(
        "--max-windows-per-segment",
        type=int,
        default=25,
        help="Cap windows per segment when balance=segment_cap",
    )
    window_parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test fraction per split",
    )
    window_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/demo",
        help="Output directory for demo artifacts",
    )
    window_parser.add_argument(
        "--report-path",
        type=str,
        default="outputs/demo/report.md",
        help="Path to write the demo report (default: outputs/demo/report.md)",
    )
    window_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run the scripted demo without an LLM (no API key required)",
    )
    window_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model override for LLM demo",
    )
    window_parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the demo report in the CLI output",
    )
    window_parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation",
    )
    _add_output_args(window_parser)

    forecasting_parser = demo_sub.add_parser(
        "forecasting",
        help="Run the forecasting comparison demo",
    )
    forecasting_parser.add_argument(
        "--run-id",
        type=str,
        default="Re200Rm200",
        help="Run ID to forecast (default: Re200Rm200)",
    )
    forecasting_parser.add_argument(
        "--variable",
        type=str,
        default="bx001_real",
        help="Variable name to forecast (default: bx001_real)",
    )
    forecasting_parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon (default: 1, chosen for tiny built-in test dataset)",
    )
    forecasting_parser.add_argument(
        "--validation-size",
        type=int,
        default=None,
        help="Validation size for holdout comparison (default: horizon)",
    )
    forecasting_parser.add_argument(
        "--methods",
        type=str,
        default="arima,theta",
        help="Comma-separated methods to compare (default: arima,theta)",
    )
    data_group = forecasting_parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--use-test-data",
        action="store_true",
        help="Force use of test data",
    )
    data_group.add_argument(
        "--full-data",
        action="store_true",
        help="Force use of full dataset",
    )
    forecasting_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/demo",
        help="Output directory for demo artifacts",
    )
    forecasting_parser.add_argument(
        "--report-path",
        type=str,
        default="outputs/demo/forecasting_report.md",
        help="Path to write the forecasting demo report (default: outputs/demo/forecasting_report.md)",
    )
    forecasting_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run the scripted forecasting demo without an LLM (no API key required)",
    )
    forecasting_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model override for LLM report generation",
    )
    forecasting_parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the demo report in the CLI output",
    )
    forecasting_parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation",
    )
    _add_output_args(forecasting_parser)

def build_parser(*, exit_on_error: bool = True) -> argparse.ArgumentParser:
    parser = TSArgumentParser(
        prog="ts-agents",
        description="CLI-first time-series workflows, tools, skills, and sandboxes",
        exit_on_error=exit_on_error,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_capabilities_subcommand(subparsers)
    _add_workflow_subcommands(subparsers)
    _add_tool_subcommands(subparsers)
    _add_sandbox_subcommands(subparsers)
    _add_skills_subcommands(subparsers)
    _add_agent_subcommands(subparsers)
    _add_data_subcommands(subparsers)
    _add_run_subcommands(subparsers)
    _add_demo_subcommands(subparsers)

    return parser


def _resolve_use_test_data(args: argparse.Namespace) -> Optional[bool]:
    if getattr(args, "use_test_data", False):
        return True
    if getattr(args, "full_data", False):
        return False
    return None


def _handle_data_command(args: argparse.Namespace) -> Tuple[Any, str]:
    from ts_agents import data_access

    data_type = args.data_type
    use_test_data = _resolve_use_test_data(args)

    runs_flag = getattr(args, "runs", False)
    vars_flag = getattr(args, "variables", False)

    show_runs = runs_flag or (args.data_command == "list" and not vars_flag)
    show_vars = vars_flag or args.data_command == "vars" or (
        args.data_command == "list" and not runs_flag
    )

    runs = data_access.list_runs(data_type=data_type, use_test_data=use_test_data) if show_runs else []
    variables = (
        data_access.list_variables(data_type=data_type, use_test_data=use_test_data)
        if show_vars
        else []
    )

    result = {
        "data_type": data_type,
        "use_test_data": use_test_data,
        "runs": runs,
        "variables": variables,
    }

    text_lines = [f"Data type: {data_type}"]
    if use_test_data is not None:
        text_lines.append(f"Use test data: {use_test_data}")
    if show_runs:
        text_lines.append("Runs:")
        text_lines.extend([f"- {run}" for run in runs])
    if show_vars:
        text_lines.append("Variables:")
        text_lines.extend([f"- {var}" for var in variables])

    return result, "\n".join(text_lines)


def _tool_supports_bundled_run(tool: Any) -> bool:
    param_names = {param.name for param in tool.parameters}
    run_names = {"unique_id", "run_id"}
    variable_names = {"variable_name", "variable"}
    return bool(param_names & run_names) and bool(param_names & variable_names)


def _tool_input_modes(tool: Any) -> List[str]:
    modes = ["param_flags", "input_json", "stdin_json"]
    if _tool_supports_bundled_run(tool):
        modes.append("bundled_run_shorthand")
    return modes


def _tool_cli_templates(tool: Any) -> List[str]:
    required = [param.name for param in tool.parameters if not param.optional]
    param_types = {param.name: param.type for param in tool.parameters}
    templates = [f"ts-agents tool show {tool.name} --json"]
    templates.append(
        normalize_cli_template(
            _build_run_example_command(tool.name, required, param_types)
        )
        + " --json"
    )
    unique_templates: List[str] = []
    for template in templates:
        if template not in unique_templates:
            unique_templates.append(template)
    return unique_templates


def _tool_status_contract(tool: Any) -> Dict[str, Any]:
    if tool.name.endswith("_with_data"):
        return {
            "result_shape": "ToolPayload wrapper",
            "status_field": "Inspect `result.status` for `ok` vs `degraded` outcomes.",
            "warnings_field": "Inspect `result.warnings` for non-fatal caveats.",
            "artifact_field": "Inspect `result.artifacts` for generated files.",
        }

    return {
        "result_shape": "Core analysis result",
        "status_field": "Core tool results do not use a standard nested status field.",
        "warnings_field": None,
        "artifact_field": None,
    }


def _tool_summary_dict(tool: Any) -> Dict[str, Any]:
    from ts_agents.tools.registry import tool_availability

    return {
        "name": tool.name,
        "category": tool.category.value,
        "cost": tool.cost.value,
        "description": tool.description,
        "dependencies": tool.dependencies,
        "optional_dependencies": tool.optional_dependencies,
        "availability": tool_availability(tool),
        "writes_artifacts": bool(tool.artifact_kinds),
        "artifact_kinds": list(tool.artifact_kinds),
    }


def _tool_detail_dict(tool: Any) -> Dict[str, Any]:
    from ts_agents.tools.registry import (
        dependency_required_extras,
        tool_availability,
        tool_dependency_details,
    )

    required = [param.name for param in tool.parameters if not param.optional]
    required_extras = sorted(
        {
            extra
            for dependency in tool.dependencies
            for extra in dependency_required_extras(dependency)
        }
    )
    availability = tool_availability(tool)
    return {
        "name": tool.name,
        "category": tool.category.value,
        "cost": tool.cost.value,
        "description": tool.description,
        "signature": tool.get_signature(),
        "dependencies": tool.dependencies,
        "optional_dependencies": tool.optional_dependencies,
        "dependency_details": tool_dependency_details(tool),
        "required_extras": required_extras,
        "availability": availability,
        "install_hint": availability.get("install_hint"),
        "parameters": [
            {
                "name": param.name,
                "type": param.type,
                "description": param.description,
                "optional": param.optional,
                "default": param.default,
            }
            for param in tool.parameters
        ],
        "required_parameters": required,
        "input_schema": tool.to_schema(),
        "returns": {
            "description": tool.returns,
        },
        "examples": tool.examples,
        "input_modes": _tool_input_modes(tool),
        "cli_templates": _tool_cli_templates(tool),
        "writes_artifacts": bool(tool.artifact_kinds),
        "artifact_kinds": list(tool.artifact_kinds),
        "status_contract": _tool_status_contract(tool),
        "resources": {
            "timeout_seconds": tool.timeout_seconds,
            "memory_mb": tool.memory_mb,
            "disk_mb": tool.disk_mb,
        },
    }


def _workflow_summary_dict(workflow: Any) -> Dict[str, Any]:
    from ts_agents.workflows import workflow_to_dict

    details = workflow_to_dict(workflow)
    return {
        "name": details["name"],
        "description": details["description"],
        "supported_input_modes": details["supported_input_modes"],
        "required_extras": details["required_extras"],
        "availability": details["availability"],
    }


def _capabilities_status_contract() -> Dict[str, Any]:
    return {
        "cli_envelope": {
            "ok_field": "True means the CLI command itself executed successfully.",
            "error_field": "When `ok` is false, inspect `error.code`, `error.message`, and `error.hint`.",
            "execution_field": "Inspect `execution` for backend and runtime metadata when a command executes work.",
            "quality_fields": "Inspect `quality_status`, `degraded`, and `requires_review` to distinguish clean success from degraded-but-executed results.",
        },
        "workflow_results": {
            "status_field": "Inspect `result.status` for `ok` vs `degraded` workflow outcomes.",
            "warnings_field": "Inspect `result.warnings` for non-fatal caveats.",
            "quality_flags_path": "If present, inspect `result.data.quality_flags` for workflow-specific quality issues.",
        },
        "tool_results": {
            "core_tools": "Core tool results return the raw analysis payload.",
            "wrapper_tools": "Many `_with_data` tools return a ToolPayload with `status`, `warnings`, `artifacts`, and `provenance`.",
        },
    }


def _handle_capabilities_command(args: argparse.Namespace) -> Tuple[Any, str]:
    from ts_agents.tools.executor import list_sandbox_backends
    from ts_agents.tools.registry import ToolRegistry, tool_availability
    from ts_agents.workflows import list_workflows, workflow_to_dict

    workflows = [workflow_to_dict(workflow) for workflow in list_workflows()]
    tools = ToolRegistry.list_all()
    available_tools = sum(1 for tool in tools if tool_availability(tool)["available"])
    tools_by_category = ToolRegistry.get_tools_for_category_summary()
    install_profile = _detect_install_profile()

    result = {
        "entrypoints": {
            "workflow_discovery": [
                "ts-agents workflow list --json",
                "ts-agents workflow show <workflow> --json",
                "ts-agents workflow run <workflow> ... --json",
            ],
            "tool_discovery": [
                "ts-agents tool search <query> --json",
                "ts-agents tool show <tool> --json",
                "ts-agents tool run <tool> ... --json",
            ],
            "skills": [
                "ts-agents skills list --json",
                "ts-agents skills show <skill> --json",
            ],
            "sandboxes": [
                "ts-agents sandbox list --json",
                "ts-agents sandbox doctor <backend> --json",
            ],
        },
        "install_profile": install_profile,
        "status_contract": _capabilities_status_contract(),
        "recommended_entrypoints": [
            "Start with `workflow list --json` for public workflows.",
            "Use `workflow show --json` before `workflow run` to inspect availability, options, artifacts, and source contracts.",
            "Use `tool search --json` and `tool show --json` for lower-level execution.",
            "Always inspect top-level quality fields plus `result.status`, `warnings`, and workflow `quality_flags` rather than relying on exit code 0 alone.",
        ],
        "workflows": workflows,
        "tools": {
            "count": len(tools),
            "available_count": available_tools,
            "unavailable_count": len(tools) - available_tools,
            "categories": tools_by_category,
        },
        "sandboxes": {
            "default_backend": os.environ.get("TS_AGENTS_SANDBOX_MODE", "local"),
            "fallback_default": "local",
            "fallback_requires_opt_in": True,
            "backends": list_sandbox_backends(),
        },
    }

    lines = [
        "CLI capabilities:",
        "- Start with workflow discovery for public automation surfaces.",
        f"- Workflows: {len(workflows)}",
        f"- Tools: {len(tools)} total, {available_tools} currently available",
        f"- Install profile: {install_profile['current_profile']}",
        f"- Default sandbox backend: {result['sandboxes']['default_backend']}",
    ]
    return result, "\n".join(lines)


def _handle_tool_command(args: argparse.Namespace) -> Tuple[Any, str]:
    from ts_agents.tools.registry import ToolRegistry, ToolCategory, ComputationalCost
    from ts_agents.tools.bundles import get_bundle_names, get_bundle_summary

    def _parse_category(raw: Optional[str]) -> Optional[Any]:
        if not raw:
            return None
        try:
            return ToolCategory(raw)
        except ValueError:
            return ToolCategory(raw.lower())

    def _parse_cost(raw: Optional[str]) -> Optional[Any]:
        if not raw:
            return None
        try:
            return ComputationalCost(raw)
        except ValueError:
            return ComputationalCost(raw.lower())

    if args.tool_command == "list":
        bundle = args.bundle
        if bundle:
            tool_names = get_bundle_names(bundle)
            tools = [ToolRegistry.get(name) for name in tool_names]
        else:
            tools = ToolRegistry.list_all()

        category = _parse_category(args.category)
        if category is not None:
            tools = [tool for tool in tools if tool.category == category]

        max_cost = _parse_cost(args.max_cost)
        if max_cost is not None:
            allowed = {tool.name for tool in ToolRegistry.list_by_max_cost(max_cost)}
            tools = [tool for tool in tools if tool.name in allowed]

        result = {
            "bundle": bundle,
            "bundle_summary": get_bundle_summary(),
            "tools": [_tool_summary_dict(tool) for tool in tools],
        }

        lines = ["Bundles:"]
        for name, info in result["bundle_summary"].items():
            lines.append(f"- {name}: {info['count']} tools")

        lines.append("Tools:")
        for tool in tools:
            dependency_hint = ""
            if tool.dependencies:
                dependency_hint = f" deps={','.join(tool.dependencies)}"
            lines.append(f"- {tool.name} ({tool.category.value}, {tool.cost.value}){dependency_hint}")

        return result, "\n".join(lines)

    if args.tool_command == "show":
        try:
            tool = ToolRegistry.get(args.tool)
        except KeyError:
            _raise_unknown_tool_error(args.tool)
        result = _tool_detail_dict(tool)
        availability = result["availability"]
        install_hint = availability.get("install_hint") or result.get("install_hint")
        lines = [
            f"Tool: {tool.name}",
            f"Category: {tool.category.value}",
            f"Cost: {tool.cost.value}",
            f"Description: {tool.description}",
            f"Returns: {tool.returns}",
            f"Availability: {availability.get('status', 'available')}",
        ]
        if tool.dependencies:
            lines.append(f"Dependencies: {', '.join(tool.dependencies)}")
        if tool.optional_dependencies:
            lines.append(f"Optional dependencies: {', '.join(tool.optional_dependencies)}")
        if result.get("required_extras"):
            lines.append(f"Required extras: {', '.join(result['required_extras'])}")
        if install_hint:
            lines.append(f"Install hint: {install_hint}")
        optional_features = availability.get("optional_features") or []
        if optional_features:
            lines.append("Optional features:")
            for feature in optional_features:
                state = "available" if feature.get("available") else "unavailable"
                extras = feature.get("required_extras") or []
                extras_suffix = f" extras={','.join(extras)}" if extras else ""
                note = feature.get("note")
                detail = f": {note}" if note else ""
                lines.append(f"- {feature['name']} [{state}]{extras_suffix}{detail}")
        if tool.parameters:
            lines.append("Parameters:")
            for param in tool.parameters:
                optional = "optional" if param.optional else "required"
                lines.append(f"- {param.name} ({param.type}, {optional})")
        lines.append("Input modes: " + ", ".join(result["input_modes"]))
        if result["writes_artifacts"]:
            lines.append("Artifacts: " + ", ".join(result["artifact_kinds"]))
        if result["cli_templates"]:
            lines.append("CLI templates:")
            for template in result["cli_templates"]:
                lines.append(f"- {template}")
        if tool.examples:
            lines.append("Examples:")
            for example in tool.examples:
                lines.append(f"- {example}")
        return result, "\n".join(lines)

    if args.tool_command == "search":
        category = _parse_category(args.category)
        max_cost = _parse_cost(args.max_cost)
        tools = ToolRegistry.search(args.query, category=category, max_cost=max_cost)
        result = {
            "query": args.query,
            "tools": [_tool_summary_dict(tool) for tool in tools],
        }
        lines = [f"Matches for '{args.query}':"]
        for tool in tools:
            lines.append(f"- {tool.name} ({tool.category.value}, {tool.cost.value})")
        if not tools:
            lines.append("(no matches)")
        return result, "\n".join(lines)

    if args.tool_command == "run":
        return _handle_run_command(args)

    raise ValueError(f"Unknown tool command: {args.tool_command}")


def _handle_run_command(args: argparse.Namespace) -> Tuple[Any, Optional[str]]:
    from ts_agents.cli.input_parsing import load_tool_params_from_json
    from ts_agents.tools.registry import ToolRegistry
    from ts_agents.tools.executor import ExecutionContext, execute_tool

    try:
        metadata = ToolRegistry.get(args.tool)
    except KeyError:
        _raise_unknown_tool_error(args.tool)

    param_types = {param.name: param.type for param in metadata.parameters}
    required = [param.name for param in metadata.parameters if not param.optional]

    input_params, input_source = load_tool_params_from_json(
        input_json=getattr(args, "input_json", None),
        use_stdin=getattr(args, "stdin", False),
        param_names=list(param_types.keys()),
    )
    params = dict(input_params)
    params.update(_parse_param_entries(args.param, param_types))
    params = _apply_run_var_args(params, param_types, args.run_id, args.variable)
    allow_fallback = getattr(args, "allow_fallback", False)
    fallback_backend = getattr(args, "fallback_backend", None) or "local"
    args._ts_input_payload = {
        "params": params,
        "input_source": input_source,
        "run_id": args.run_id,
        "variable": args.variable,
        "sandbox": args.sandbox or os.environ.get("TS_AGENTS_SANDBOX_MODE") or "local",
        "allow_fallback": allow_fallback,
        "fallback_backend": fallback_backend,
    }
    if getattr(args, "fallback_backend", None) and not allow_fallback:
        raise ValueError(
            "--fallback-backend requires --allow-fallback to take effect. "
            "Pass --allow-fallback to opt in to backend fallback."
        )
    _raise_missing_required_error(
        tool_name=args.tool,
        param_types=param_types,
        required=required,
        provided=params,
    )

    sandbox_mode = args.sandbox or os.environ.get("TS_AGENTS_SANDBOX_MODE")
    context = ExecutionContext(
        sandbox_mode=sandbox_mode,
        user_approved=getattr(args, "approve", False),
        allow_network=getattr(args, "allow_network", False),
        allow_fallback=allow_fallback,
        fallback_backend=fallback_backend,
    )
    execution = execute_tool(args.tool, params, context=context)
    args._ts_execution_result = execution
    if not execution.success:
        if execution.error:
            raise execution.error
        raise RuntimeError(execution.formatted_output or "Tool execution failed")

    return execution.result, execution.formatted_output or None


def _handle_workflow_command(args: argparse.Namespace) -> Tuple[Any, Optional[str]]:
    from ts_agents.tools.executor import ExecutionContext
    from ts_agents.workflows import get_workflow, list_workflows, workflow_to_dict
    from ts_agents.workflows.executor import execute_workflow

    if args.workflow_command == "list":
        workflows = list_workflows()
        result = {
            "workflows": [_workflow_summary_dict(workflow) for workflow in workflows]
        }
        lines = ["Workflows:"]
        for workflow in workflows:
            availability = workflow.availability()
            status = availability.get("status", "available")
            hint = ""
            if availability.get("install_hint"):
                hint = f" ({availability['install_hint']})"
            lines.append(f"- {workflow.name} [{status}]: {workflow.description}{hint}")
        return result, "\n".join(lines)

    if args.workflow_command == "show":
        try:
            workflow = get_workflow(args.workflow_name)
        except KeyError:
            _raise_unknown_workflow_error(args.workflow_name)
        result = workflow_to_dict(workflow)
        availability = result["availability"]
        lines = [
            f"Workflow: {result['name']}",
            f"Description: {result['description']}",
            f"Availability: {availability.get('status', 'available')}",
            f"Input modes: {', '.join(result['supported_input_modes'])}",
            f"Source requirement: {result['source_requirement']}",
        ]
        if result["required_extras"]:
            lines.append(f"Required extras: {', '.join(result['required_extras'])}")
        if availability.get("missing_dependencies"):
            lines.append(
                "Missing dependencies: " + ", ".join(availability["missing_dependencies"])
            )
        if availability.get("install_hint"):
            lines.append(f"Install hint: {availability['install_hint']}")
        if result["options"]:
            lines.append("Options:")
            for option in result["options"]:
                optionality = "required" if option["required"] else "optional"
                lines.append(f"- {option['name']} ({option['type']}, {optionality})")
        if result["source_options"]:
            lines.append("Source options:")
            for option in result["source_options"]:
                optionality = "required" if option["required"] else "optional"
                lines.append(f"- {option['name']} ({option['type']}, {optionality})")
        if result["global_options"]:
            lines.append("Global options:")
            for option in result["global_options"]:
                lines.append(f"- {option['name']} ({option['type']})")
        if result["artifacts"]:
            lines.append("Artifacts:")
            for artifact in result["artifacts"]:
                requirement = "required" if artifact["required"] else "optional"
                lines.append(f"- {artifact['filename']} ({requirement})")
        default_output_behavior = result.get("default_output_behavior") or {}
        if default_output_behavior:
            lines.append("Output behavior:")
            if default_output_behavior.get("default_output_dir"):
                lines.append(f"- Default root: {default_output_behavior['default_output_dir']}")
            if default_output_behavior.get("manifest_filename"):
                lines.append(f"- Manifest: {default_output_behavior['manifest_filename']}")
            if default_output_behavior.get("behavior"):
                lines.append(f"- {default_output_behavior['behavior']}")
            if default_output_behavior.get("collision_policy"):
                lines.append(f"- {default_output_behavior['collision_policy']}")
        if result["cli_templates"]:
            lines.append("CLI templates:")
            for template in result["cli_templates"]:
                lines.append(f"- {template}")
        if result["examples"]:
            lines.append("Examples:")
            for example in result["examples"]:
                lines.append(f"- {example}")
        return result, "\n".join(lines)

    if args.workflow_command != "run":
        raise ValueError(f"Unknown workflow command: {args.workflow_command}")

    try:
        workflow = get_workflow(args.workflow_name)
    except KeyError:
        _raise_unknown_workflow_error(args.workflow_name)
    _validate_workflow_source_args(args)
    args.use_test_data_resolved = _resolve_use_test_data(args)
    run_lifecycle = _prepare_workflow_run_output(args, workflow)

    workflow_input = workflow.load_input(args)
    runner_kwargs = workflow.build_runner_kwargs(args)
    runner_kwargs.update(
        {
            "run_id": run_lifecycle["run_id"],
            "resumed": run_lifecycle["resumed"],
            "output_dir_mode": run_lifecycle["output_dir_mode"],
        }
    )
    public_run_lifecycle = {
        key: value
        for key, value in run_lifecycle.items()
        if not key.startswith("_")
    }
    allow_fallback = getattr(args, "allow_fallback", False)
    fallback_backend = getattr(args, "fallback_backend", None) or "local"
    args._ts_input_payload = {
        "workflow": args.workflow_name,
        "source": _workflow_input_source_ref(workflow_input),
        "options": runner_kwargs,
        "run": public_run_lifecycle,
        "sandbox": getattr(args, "sandbox", None)
        or os.environ.get("TS_AGENTS_SANDBOX_MODE")
        or "local",
        "allow_fallback": allow_fallback,
        "fallback_backend": fallback_backend,
    }
    if getattr(args, "fallback_backend", None) and not allow_fallback:
        raise ValueError(
            "--fallback-backend requires --allow-fallback to take effect. "
            "Pass --allow-fallback to opt in to backend fallback."
        )

    _materialize_workflow_output_dir(run_lifecycle)

    sandbox_mode = getattr(args, "sandbox", None) or os.environ.get("TS_AGENTS_SANDBOX_MODE")
    context = ExecutionContext(
        sandbox_mode=sandbox_mode,
        allow_network=getattr(args, "allow_network", False),
        allow_fallback=allow_fallback,
        fallback_backend=fallback_backend,
    )
    execution = execute_workflow(
        args.workflow_name,
        workflow_input,
        runner_kwargs,
        context=context,
    )
    args._ts_execution_result = execution
    if not execution.success:
        if execution.error:
            raise execution.error
        raise RuntimeError(execution.formatted_output or "Workflow execution failed")

    _synchronize_workflow_manifest(execution.result, execution)
    return execution.result, execution.formatted_output or None


def _validate_workflow_source_args(args: argparse.Namespace) -> None:
    run_id = getattr(args, "run_id", None)
    variable = getattr(args, "variable", None)
    if bool(run_id) != bool(variable):
        raise ValueError("Bundled workflow inputs require both --run-id and --variable.")

    if variable and any(
        (
            getattr(args, "input", None),
            getattr(args, "input_json", None),
            getattr(args, "stdin", False),
        )
    ):
        raise ValueError(
            "--variable is only valid with --run-id and cannot be combined with --input, --input-json, or --stdin."
        )


def _workflow_input_source_ref(workflow_input: Any) -> Dict[str, Any]:
    provenance = getattr(workflow_input, "provenance", {}) or {}
    return (
        provenance.get("input_ref")
        or provenance.get("series_ref")
        or provenance.get("stream_ref")
        or {}
    )


def _approval_prompt(tool_name: str) -> bool:
    response = input(f"Approve operation '{tool_name}'? [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def _handle_agent_command(args: argparse.Namespace) -> Tuple[Any, Optional[str]]:
    if args.type == "simple":
        from ts_agents.agents.simple.agent import run_single_query

        response = run_single_query(
            query=args.prompt,
            tool_bundle=args.tool_bundle,
            model_name=args.model,
        )
        return {"response": response}, response

    from ts_agents.agents.deep.orchestrator import create_deep_agent, run_with_approval

    enable_approval = args.approval != "off"
    agent = create_deep_agent(
        model_name=args.model,
        enable_approval=enable_approval,
        enable_logging=False,
    )

    callback = None
    if args.approval == "auto":
        callback = lambda tool_name: True
    elif args.approval == "prompt":
        callback = _approval_prompt

    response = run_with_approval(agent, args.prompt, approval_callback=callback)
    return {"response": response}, response


def _handle_sandbox_command(args: argparse.Namespace) -> Tuple[Any, str]:
    from ts_agents.tools.executor import describe_sandbox_backend, list_sandbox_backends

    if args.sandbox_command == "list":
        result = {
            "default_backend": os.environ.get("TS_AGENTS_SANDBOX_MODE", "local"),
            "fallback_default": "local",
            "fallback_requires_opt_in": True,
            "backends": list_sandbox_backends(),
        }
        lines = [
            f"Default backend: {result['default_backend']}",
            "Fallback requires --allow-fallback (default backend: local).",
            "Backends:",
        ]
        for backend in result["backends"]:
            state = "available" if backend["available"] else "unavailable"
            line = f"- {backend['backend']}: {state}"
            if backend.get("reason"):
                line += f" ({backend['reason']})"
            lines.append(line)
        return result, "\n".join(lines)

    if args.sandbox_command == "doctor":
        result = describe_sandbox_backend(args.backend)
        lines = [
            f"Backend: {result['backend']}",
            f"Available: {result['available']}",
            f"Description: {result['description']}",
        ]
        if result.get("reason"):
            lines.append(f"Reason: {result['reason']}")
        if result.get("suggested_fix"):
            lines.append(f"Suggested fix: {result['suggested_fix']}")
        if result.get("requirements"):
            lines.append("Requirements:")
            for requirement in result["requirements"]:
                lines.append(f"- {requirement}")
        return result, "\n".join(lines)

    raise ValueError(f"Unknown sandbox command: {args.sandbox_command}")


def _handle_skills_command(args: argparse.Namespace) -> Tuple[Any, str]:
    from pathlib import Path
    from .skills import (
        export_skills,
        get_skill_details,
        validate_all_skills,
        list_skills,
        parse_skill_frontmatter,
    )

    if args.skills_command == "export":
        output_path = export_skills(
            args.out,
            agent=getattr(args, "agent", None),
            all_agents=getattr(args, "all_agents", False),
            use_symlinks=getattr(args, "symlink", False),
            format_name=getattr(args, "format", None),
        )
        result = {
            "skills_path": str(output_path),
            "format": getattr(args, "format", None) or ("json" if str(output_path).endswith(".json") else "markdown"),
        }
        if getattr(args, "all_agents", False):
            mode = "symlinks" if getattr(args, "symlink", False) else "copies"
            return result, f"Exported skills as {mode} to all agent directories under {output_path}"
        elif getattr(args, "agent", None):
            mode = "symlinks" if getattr(args, "symlink", False) else "copies"
            return result, f"Exported skills as {mode} to {output_path}"
        if result["format"] == "json":
            return result, f"Exported structured skills metadata to {output_path}"
        return result, f"Exported skills to {output_path}"

    elif args.skills_command == "validate":
        skills_dir = Path(args.path) if args.path else None
        errors = validate_all_skills(skills_dir)

        if not errors:
            result = {"valid": True, "errors": {}}
            return result, "All skills validated successfully"
        else:
            result = {"valid": False, "errors": errors}
            lines = ["Validation errors found:"]
            for skill_name, skill_errors in errors.items():
                lines.append(f"\n{skill_name}:")
                for error in skill_errors:
                    lines.append(f"  - {error}")
            return result, "\n".join(lines)

    elif args.skills_command == "list":
        skills_dir = Path(args.path) if args.path else None
        skill_dirs = list_skills(skills_dir)

        skills_info = []
        for skill_dir in skill_dirs:
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                frontmatter, _ = parse_skill_frontmatter(skill_file)
                skills_info.append({
                    "name": skill_dir.name,
                    "description": frontmatter.get("description", ""),
                })

        result = {"skills": skills_info}
        lines = ["Available skills:"]
        for info in skills_info:
            desc = info["description"][:60] + "..." if len(info["description"]) > 60 else info["description"]
            lines.append(f"  - {info['name']}: {desc}")
        return result, "\n".join(lines)

    elif args.skills_command == "show":
        skills_dir = Path(args.path) if args.path else None
        result = get_skill_details(args.skill, skills_dir=skills_dir)
        lines = [
            f"Skill: {result['name']}",
            f"Description: {result['description']}",
        ]
        preferred_workflow = ((result.get("metadata") or {}).get("ts_agents") or {}).get("preferred_workflow")
        if preferred_workflow:
            lines.append(f"Preferred workflow: {preferred_workflow}")
        preferred_tools = ((result.get("metadata") or {}).get("ts_agents") or {}).get("preferred_tools") or []
        if preferred_tools:
            lines.append(f"Preferred tools: {', '.join(preferred_tools)}")
        if result.get("commands"):
            lines.append("Commands:")
            for command in result["commands"][:5]:
                lines.append(f"- {command}")
        return result, "\n".join(lines)

    else:
        raise ValueError(f"Unknown skills command: {args.skills_command}")


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _split_int_csv(raw: str) -> List[int]:
    return [int(part) for part in _split_csv(raw)]


def _find_tool_payload(tool_calls: List[Any], tool_name: str) -> Optional[Dict[str, Any]]:
    for call in reversed(tool_calls):
        if call.tool_name == tool_name and call.result_payload is not None:
            return call.result_payload
    return None


def _resolve_demo_report_path(
    args: argparse.Namespace,
    default_report_path: str = "outputs/demo/report.md",
):
    from pathlib import Path

    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    default_report_path_obj = Path(default_report_path)

    if report_path == default_report_path_obj and output_dir != default_report_path_obj.parent:
        return output_dir / default_report_path_obj.name
    return report_path


def _format_metric_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "N/A"


def _build_demo_window_report(
    *,
    csv_path: str,
    classifier: str,
    selection_payload: Dict[str, Any],
    eval_payload: Dict[str, Any],
) -> str:
    classification = eval_payload.get("classification") or {}

    best_window = selection_payload.get("best_window_size")
    metric = eval_payload.get("metric", "metric")
    score = eval_payload.get("score")
    accuracy = classification.get("accuracy")
    f1_macro = classification.get("f1_score")
    classifier_used = classification.get("method") or eval_payload.get("classifier") or classifier

    notes: List[str] = []
    confusion = classification.get("confusion_matrix")
    if isinstance(confusion, list):
        active_true_classes = 0
        for row in confusion:
            if isinstance(row, list) and any(v != 0 for v in row):
                active_true_classes += 1
        if active_true_classes <= 1:
            notes.append("Test split appears single-class.")

    if (
        isinstance(score, (int, float))
        and isinstance(accuracy, (int, float))
        and isinstance(f1_macro, (int, float))
        and score == 1.0
        and accuracy == 1.0
        and f1_macro == 1.0
    ):
        notes.append("All reported metrics are perfect; verify split balance and leakage assumptions.")

    note_text = " ".join(notes) if notes else "No obvious pathologies detected in reported metrics."

    lines = [
        "### Report on Windowed Classification",
        "",
        f"- **Dataset Path**: `{csv_path}`",
        f"- **Classifier Used**: {classifier_used}",
        f"- **Best Window Size**: {best_window}",
        f"- **Metric + Score**: {metric} - {_format_metric_value(score)}",
        f"- **Accuracy**: {_format_metric_value(accuracy)}",
        f"- **Macro F1**: {_format_metric_value(f1_macro)}",
        "",
        note_text,
    ]
    return "\n".join(lines)


def _build_demo_forecasting_report(
    *,
    run_id: str,
    variable_name: str,
    horizon: int,
    methods: List[str],
    comparison_payload: Dict[str, Any],
) -> str:
    rankings = comparison_payload.get("rankings") or {}
    rmse_ranking = rankings.get("rmse") or []
    best_method = rmse_ranking[0] if rmse_ranking else "N/A"
    recommendation = comparison_payload.get("recommendation") or "No recommendation generated."

    method_lines: List[str] = []
    metrics = comparison_payload.get("metrics") or {}
    for method in methods:
        method_metric = metrics.get(method)
        if not isinstance(method_metric, dict):
            method_lines.append(f"- `{method}`: no metrics available")
            continue
        if "error" in method_metric:
            method_lines.append(f"- `{method}`: error - {method_metric['error']}")
            continue
        method_lines.append(
            f"- `{method}`: RMSE={_format_metric_value(method_metric.get('rmse'))}, "
            f"MAE={_format_metric_value(method_metric.get('mae'))}, "
            f"MAPE={_format_metric_value(method_metric.get('mape'))}%"
        )

    lines = [
        "### Report on Forecasting Demo",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Variable**: `{variable_name}`",
        f"- **Horizon**: {horizon}",
        f"- **Compared Methods**: {', '.join(methods)}",
        f"- **Best Method (RMSE)**: {best_method}",
        "",
        "#### Metrics",
        *method_lines,
        "",
        "#### Recommendation",
        recommendation,
    ]
    return "\n".join(lines)


def _run_forecasting_demo_comparison(args: argparse.Namespace):
    from pathlib import Path
    import warnings

    from ts_agents import data_access
    from ts_agents.cli.output import render_output, to_jsonable, write_output
    from ts_agents.core.comparison import compare_forecasting_methods, plot_forecast_comparison

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_test_data = _resolve_use_test_data(args)
    methods = _split_csv(args.methods) if args.methods else ["arima", "theta"]
    if not methods:
        methods = ["arima", "theta"]

    series = data_access.get_series(
        run_id=args.run_id,
        variable_name=args.variable,
        use_test_data=use_test_data,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comparison = compare_forecasting_methods(
            series,
            horizon=args.horizon,
            methods=methods,
            validation_size=args.validation_size,
        )

    comparison_payload = to_jsonable(comparison)
    comparison_path = output_dir / "forecast_comparison.json"
    write_output(render_output(comparison_payload, json_output=True), str(comparison_path))

    plot_path = output_dir / "forecast_comparison.png"
    if not args.skip_plots:
        fig = plot_forecast_comparison(comparison, series)
        fig.savefig(str(plot_path), format="png")
        plot_lib = _get_plot_lib()
        plot_lib.close(fig)

    rmse_ranking = (comparison_payload.get("rankings") or {}).get("rmse") or []
    best_method = rmse_ranking[0] if rmse_ranking else None

    return {
        "series": series,
        "methods": methods,
        "comparison_payload": comparison_payload,
        "comparison_path": comparison_path,
        "plot_path": plot_path,
        "best_method": best_method,
        "output_dir": output_dir,
    }


def _get_plot_lib():
    import matplotlib.pyplot as plt

    return plt


def _render_forecasting_report_with_llm(
    *,
    model_name: Optional[str],
    run_id: str,
    variable_name: str,
    horizon: int,
    methods: List[str],
    comparison_payload: Dict[str, Any],
) -> str:
    from langchain_openai import ChatOpenAI
    from ts_agents.config import get_openai_model

    llm = ChatOpenAI(model=model_name or get_openai_model(), temperature=0)
    prompt = (
        "Write a concise markdown report for a forecasting demo.\n"
        "Use <= 14 lines and include: run_id, variable, horizon, compared methods, "
        "best method by RMSE (if available), key metrics, and one caveat.\n\n"
        f"run_id: {run_id}\n"
        f"variable: {variable_name}\n"
        f"horizon: {horizon}\n"
        f"methods: {methods}\n"
        f"comparison_json: {dump_json(comparison_payload, indent=None)}"
    )
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    return str(content)


def _generate_demo_data(args: argparse.Namespace, csv_path: Path) -> None:
    import subprocess

    if args.no_generate:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    synth_script = _resolve_required_runtime_path("data/make_synthetic_labeled_stream.py")
    cmd = [
        sys.executable,
        str(synth_script),
        "--scenario",
        args.scenario,
        "--hz",
        str(args.hz),
        "--minutes",
        str(args.minutes),
        "--seed",
        str(args.seed),
        "--out",
        str(csv_path),
    ]
    subprocess.run(cmd, check=True)


def _run_demo_window_classification_scripted(args: argparse.Namespace) -> Dict[str, Any]:
    from pathlib import Path

    from ts_agents.cli.input_parsing import load_labeled_stream_input
    from ts_agents.cli.output import write_output
    from ts_agents.workflows.activity import run_activity_recognition_workflow

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _generate_demo_data(args, csv_path)

    value_columns = _split_csv(args.value_columns)
    workflow_input = load_labeled_stream_input(
        input_path=str(csv_path),
        time_col=None,
        value_cols=value_columns,
        label_col=args.label_column,
    )
    payload = run_activity_recognition_workflow(
        workflow_input,
        output_dir=str(output_dir),
        window_sizes=_split_int_csv(args.window_sizes) if args.window_sizes else None,
        metric=args.metric,
        classifier=args.classifier,
        balance=args.balance,
        max_windows_per_segment=args.max_windows_per_segment,
        test_size=args.test_size,
        seed=args.seed,
        skip_plots=args.skip_plots,
    )
    selection_payload = payload.data["window_selection"]
    eval_payload = payload.data["evaluation"]
    classifier = payload.data["classifier_used"]

    report = _build_demo_window_report(
        csv_path=str(csv_path),
        classifier=classifier,
        selection_payload=selection_payload,
        eval_payload=eval_payload,
    )
    report_path = _resolve_demo_report_path(args)
    write_output(report, str(report_path))

    return {
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "best_window_size": int(payload.data["best_window_size"]),
        "metric": str(eval_payload["metric"]),
        "score": float(eval_payload["score"]),
        "classifier": classifier,
        "window_selection_path": str(output_dir / "window_selection.json"),
        "eval_path": str(output_dir / "eval.json"),
        "window_scores_plot": (
            str(output_dir / "window_scores.png")
            if (output_dir / "window_scores.png").exists()
            else None
        ),
        "confusion_matrix_plot": (
            str(output_dir / "confusion_matrix.png")
            if (output_dir / "confusion_matrix.png").exists()
            else None
        ),
        "report_path": str(report_path),
        "report": report,
        "mode": "scripted",
    }


def _run_demo_window_classification_llm(args: argparse.Namespace) -> Dict[str, Any]:
    import json
    import subprocess
    from pathlib import Path

    from ts_agents.agents.simple import SimpleAgentChat
    from ts_agents.cli.output import write_output

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is required for the LLM demo. "
            "Set OPENAI_API_KEY or run with --no-llm (scripted)."
        )

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _generate_demo_data(args, csv_path)

    value_columns = _split_csv(args.value_columns)
    window_sizes = [int(v) for v in _split_csv(args.window_sizes)] if args.window_sizes else None
    classifier_hint = args.classifier if args.classifier != "auto" else "choose one"
    window_sizes_text = window_sizes if window_sizes else "auto"

    prompt = (
        "You are running the ts-agents demo. Use tools to:\n"
        "1) select a window size with select_window_size_from_csv\n"
        "2) evaluate with evaluate_windowed_classifier_from_csv using the best window size\n\n"
        f"Dataset: {csv_path}\n"
        f"value_columns: {','.join(value_columns)} (pass as a string)\n"
        f"label_column: {args.label_column}\n"
        f"window_sizes: {window_sizes_text}\n"
        f"metric: {args.metric}\n"
        f"balance: {args.balance}\n"
        f"max_windows_per_segment: {args.max_windows_per_segment}\n"
        f"test_size: {args.test_size}\n"
        f"seed: {args.seed}\n"
        f"classifier: {classifier_hint} (minirocket|rocket|knn)\n\n"
        "After the tool calls, write a short Markdown report with:\n"
        "- Dataset path\n"
        "- Classifier used\n"
        "- Best window size\n"
        "- Metric + score\n"
        "- Accuracy and macro F1\n"
        "- A brief note if the test split appears single-class or perfect.\n"
        "Keep it concise (<12 lines)."
    )

    chat = SimpleAgentChat(
        model_name=args.model,
        tool_bundle="demo",
        enable_logging=True,
        capture_results=True,
    )
    report = chat.chat(prompt)

    tool_calls = chat.turns[-1].tool_calls if chat.turns else []
    selection_payload = _find_tool_payload(tool_calls, "select_window_size_from_csv")
    eval_payload = _find_tool_payload(tool_calls, "evaluate_windowed_classifier_from_csv")

    if selection_payload is None or eval_payload is None:
        raise ValueError(
            "LLM demo did not return expected tool outputs. "
            "Re-run or use --no-llm."
        )

    selection_path = output_dir / "window_selection.json"
    eval_path = output_dir / "eval.json"
    selection_path.write_text(dump_json(selection_payload))
    eval_path.write_text(dump_json(eval_payload))

    window_plot = output_dir / "window_scores.png"
    confusion_plot = output_dir / "confusion_matrix.png"

    if not args.skip_plots:
        plot_window_selection = _resolve_required_runtime_path("demo/plot_window_selection.py")
        plot_confusion_matrix = _resolve_required_runtime_path("demo/plot_confusion_matrix.py")
        subprocess.run(
            [
                sys.executable,
                str(plot_window_selection),
                str(selection_path),
                str(window_plot),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(plot_confusion_matrix),
                str(eval_path),
                str(confusion_plot),
            ],
            check=True,
        )

    report_path = _resolve_demo_report_path(args)
    write_output(report, str(report_path))

    return {
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "best_window_size": selection_payload.get("best_window_size"),
        "metric": eval_payload.get("metric"),
        "score": eval_payload.get("score"),
        "classifier": eval_payload.get("classifier"),
        "window_selection_path": str(selection_path),
        "eval_path": str(eval_path),
        "window_scores_plot": str(window_plot) if window_plot.exists() else None,
        "confusion_matrix_plot": str(confusion_plot) if confusion_plot.exists() else None,
        "report_path": str(report_path),
        "report": report,
        "mode": "llm",
    }


def _run_demo_forecasting_scripted(args: argparse.Namespace) -> Dict[str, Any]:
    from pathlib import Path

    from ts_agents.cli.input_parsing import load_series_input
    from ts_agents.cli.output import write_output
    from ts_agents.workflows.forecast import run_forecast_series_workflow

    series_input = load_series_input(
        run_id=args.run_id,
        variable_name=args.variable,
        use_test_data=_resolve_use_test_data(args),
    )
    payload = run_forecast_series_workflow(
        series_input,
        output_dir=args.output_dir,
        horizon=args.horizon,
        methods=_split_csv(args.methods),
        validation_size=args.validation_size,
        skip_plots=args.skip_plots,
        report_mode="scripted",
    )
    workflow_output_dir = Path(args.output_dir)
    report = _as_demo_forecasting_report((workflow_output_dir / "report.md").read_text())
    report_path = _resolve_demo_report_path(
        args,
        default_report_path="outputs/demo/forecasting_report.md",
    )
    write_output(report, str(report_path))

    return {
        "run_id": args.run_id,
        "variable": args.variable,
        "horizon": int(args.horizon),
        "methods": payload.data.get("methods", []),
        "output_dir": str(workflow_output_dir),
        "comparison_path": str(workflow_output_dir / "forecast_comparison.json"),
        "forecast_comparison_plot": (
            str(workflow_output_dir / "forecast_comparison.png")
            if (workflow_output_dir / "forecast_comparison.png").exists()
            else None
        ),
        "best_method": payload.data.get("best_method"),
        "report_path": str(report_path),
        "report": report,
        "mode": "scripted",
    }


def _run_demo_forecasting_llm(args: argparse.Namespace) -> Dict[str, Any]:
    from pathlib import Path

    from ts_agents.cli.input_parsing import load_series_input
    from ts_agents.cli.output import write_output
    from ts_agents.workflows.forecast import run_forecast_series_workflow

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is required for the LLM demo. "
            "Set OPENAI_API_KEY or run with --no-llm (scripted)."
        )

    series_input = load_series_input(
        run_id=args.run_id,
        variable_name=args.variable,
        use_test_data=_resolve_use_test_data(args),
    )
    payload = run_forecast_series_workflow(
        series_input,
        output_dir=args.output_dir,
        horizon=args.horizon,
        methods=_split_csv(args.methods),
        validation_size=args.validation_size,
        skip_plots=args.skip_plots,
        report_mode="llm",
        model_name=args.model,
    )
    workflow_output_dir = Path(args.output_dir)
    report = _as_demo_forecasting_report((workflow_output_dir / "report.md").read_text())
    report_path = _resolve_demo_report_path(
        args,
        default_report_path="outputs/demo/forecasting_report.md",
    )
    write_output(report, str(report_path))

    return {
        "run_id": args.run_id,
        "variable": args.variable,
        "horizon": int(args.horizon),
        "methods": payload.data.get("methods", []),
        "output_dir": str(workflow_output_dir),
        "comparison_path": str(workflow_output_dir / "forecast_comparison.json"),
        "forecast_comparison_plot": (
            str(workflow_output_dir / "forecast_comparison.png")
            if (workflow_output_dir / "forecast_comparison.png").exists()
            else None
        ),
        "best_method": payload.data.get("best_method"),
        "report_path": str(report_path),
        "report": report,
        "mode": "llm",
    }


def _as_demo_forecasting_report(report: str) -> str:
    return report.replace(
        "### Report on Forecast-Series Workflow",
        "### Report on Forecasting Demo",
        1,
    )


def _handle_demo_command(args: argparse.Namespace) -> Tuple[Any, str]:
    if args.demo_command == "window-classification":
        if getattr(args, "no_llm", False):
            result = _run_demo_window_classification_scripted(args)
        else:
            result = _run_demo_window_classification_llm(args)
        lines = [
            "Demo complete.",
            f"- Data: {result['csv_path']}",
            f"- Output dir: {result['output_dir']}",
            f"- Best window size: {result['best_window_size']}",
            f"- {result['metric']}: {result['score']:.4f}",
        ]
        text = "\n".join(lines)
        if getattr(args, "print_report", False) and result.get("report"):
            text = text + "\n\n" + result["report"]
        return result, text

    if args.demo_command == "forecasting":
        if getattr(args, "no_llm", False):
            result = _run_demo_forecasting_scripted(args)
        else:
            result = _run_demo_forecasting_llm(args)

        best_method_text = result["best_method"] if result["best_method"] else "N/A"
        lines = [
            "Demo complete.",
            f"- Run ID: {result['run_id']}",
            f"- Variable: {result['variable']}",
            f"- Horizon: {result['horizon']}",
            f"- Best method (RMSE): {best_method_text}",
            f"- Output dir: {result['output_dir']}",
        ]
        text = "\n".join(lines)
        if getattr(args, "print_report", False) and result.get("report"):
            text = text + "\n\n" + result["report"]
        return result, text

    raise ValueError(f"Unknown demo command: {args.demo_command}")


def _emit_deprecation_warning(args: argparse.Namespace) -> None:
    if args.command == "run":
        print(
            "Warning: `ts-agents run` is deprecated; prefer `ts-agents tool run`.",
            file=sys.stderr,
        )
        return
    if args.command == "demo":
        print(
            "Warning: `ts-agents demo` is a legacy compatibility surface; prefer `ts-agents workflow run` for new automation.",
            file=sys.stderr,
        )


def run(argv: Optional[List[str]] = None) -> int:
    from ts_agents.config import load_user_env

    load_user_env()
    json_requested = _argv_requests_json(argv)
    parser = build_parser(exit_on_error=not json_requested)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        if json_requested:
            args, unknown = parser.parse_known_args(argv)
            if unknown:
                raise argparse.ArgumentError(
                    None,
                    f"unrecognized arguments: {' '.join(unknown)}",
                )
        else:
            args = parser.parse_args(argv)
    except argparse.ArgumentError as exc:
        if json_requested:
            print(render_output(_parse_error_envelope(raw_argv, exc), json_output=True))
            return 2

        parser = build_parser()
        try:
            parser.parse_args(argv)
        except SystemExit as parse_exit:
            return int(parse_exit.code)
        raise

    args._ts_raw_argv = raw_argv
    _emit_deprecation_warning(args)

    try:
        if getattr(args, "extract_images", None) and not getattr(args, "save", None):
            raise ValueError("--extract-images requires --save")

        if args.command == "capabilities":
            result, text = _handle_capabilities_command(args)
        elif args.command == "data":
            result, text = _handle_data_command(args)
        elif args.command == "tool":
            result, text = _handle_tool_command(args)
        elif args.command == "run":
            result, text = _handle_run_command(args)
        elif args.command == "agent":
            result, text = _handle_agent_command(args)
        elif args.command == "sandbox":
            result, text = _handle_sandbox_command(args)
        elif args.command == "skills":
            result, text = _handle_skills_command(args)
        elif args.command == "workflow":
            result, text = _handle_workflow_command(args)
        elif args.command == "demo":
            result, text = _handle_demo_command(args)
        else:
            parser.error("Unknown command")
            return 2

        output_result = _success_envelope(args, result) if args.json else result
        output = render_output(output_result, json_output=args.json, text_output=text)
        print(output)

        if args.save:
            content_to_save = output
            extracted_paths: List[str] = []

            if args.extract_images:
                filename_prefix = Path(args.save).stem or "image"
                if args.json:
                    payload = to_jsonable(output_result)
                    payload, image_paths = extract_images_from_jsonable(
                        payload,
                        image_dir=args.extract_images,
                        filename_prefix=filename_prefix,
                    )
                    content_to_save = dump_json(payload)
                else:
                    content_to_save, image_paths = extract_images_to_files(
                        output,
                        image_dir=args.extract_images,
                        filename_prefix=filename_prefix,
                    )
                extracted_paths = [str(path) for path in image_paths]

            path = write_output(content_to_save, args.save)
            if not args.json:
                print(f"Saved output to {path}")
                if args.extract_images:
                    if extracted_paths:
                        print(
                            f"Extracted {len(extracted_paths)} image(s) to {args.extract_images}"
                        )
                    else:
                        print("No embedded images found to extract.")

        return 0
    except Exception as exc:
        if getattr(args, "json", False):
            error_output = render_output(_error_envelope(args, exc), json_output=True)
            print(error_output)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return _exit_code_for_exception(exc)


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
