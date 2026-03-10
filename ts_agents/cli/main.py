"""CLI entrypoint and subcommand handlers."""

from __future__ import annotations

import argparse
from difflib import get_close_matches
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Dict, List, Optional, Tuple

from .output import (
    extract_images_from_jsonable,
    extract_images_to_files,
    render_output,
    to_jsonable,
    write_output,
)


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw}")


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
    parts = [f"uv run ts-agents run {tool_name}"]

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
    data_parser = subparsers.add_parser("data", help="Data discovery commands")
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


def _add_run_subcommands(subparsers: argparse._SubParsersAction) -> None:
    run_parser = subparsers.add_parser(
        "run",
        help="Run a tool by name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run ts-agents run forecast_theta_with_data --run Re200Rm200 --var bx001_real --param horizon=12\n"
            "  uv run ts-agents run compare_forecasts_with_data --run Re200Rm200 --var bx001_real --param models=arima,theta --json\n"
            "  uv run ts-agents run stl_decompose_with_data --run Re200Rm200 --var bx001_real --save outputs/stl.md"
        ),
    )
    run_parser.add_argument("tool", type=str, help="Tool name to execute")
    run_parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Tool parameter in key=value form (repeatable)",
    )
    run_parser.add_argument(
        "--run",
        dest="run_id",
        type=str,
        default=None,
        help="Run ID (maps to unique_id/run_id)",
    )
    run_parser.add_argument(
        "--var",
        dest="variable",
        type=str,
        default=None,
        help="Variable name (maps to variable_name/variable)",
    )
    run_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve execution of very high cost tools",
    )
    run_parser.add_argument(
        "--sandbox",
        choices=["local", "subprocess", "docker", "daytona", "modal"],
        default=None,
        help="Execution sandbox (overrides TS_AGENTS_SANDBOX_MODE)",
    )
    run_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network access in sandboxed execution (if supported)",
    )
    _add_output_args(run_parser)


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
        help="Run curated demo workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ts-agents",
        description="Time series analysis CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_data_subcommands(subparsers)
    _add_tool_subcommands(subparsers)
    _add_run_subcommands(subparsers)
    _add_agent_subcommands(subparsers)
    _add_skills_subcommands(subparsers)
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


def _handle_tool_command(args: argparse.Namespace) -> Tuple[Any, str]:
    from ts_agents.tools.registry import ToolRegistry, ToolCategory, ComputationalCost
    from ts_agents.tools.bundles import get_bundle_names, get_bundle_summary

    bundle = args.bundle
    if bundle:
        tool_names = get_bundle_names(bundle)
        tools = [ToolRegistry.get(name) for name in tool_names]
    else:
        tools = ToolRegistry.list_all()

    if args.category:
        try:
            category = ToolCategory(args.category)
        except ValueError:
            category = ToolCategory(args.category.lower())
        tools = [tool for tool in tools if tool.category == category]

    if args.max_cost:
        try:
            max_cost = ComputationalCost(args.max_cost)
        except ValueError:
            max_cost = ComputationalCost(args.max_cost.lower())
        allowed = {tool.name for tool in ToolRegistry.list_by_max_cost(max_cost)}
        tools = [tool for tool in tools if tool.name in allowed]

    result = {
        "bundle": bundle,
        "bundle_summary": get_bundle_summary(),
        "tools": [
            {
                "name": tool.name,
                "category": tool.category.value,
                "cost": tool.cost.value,
                "description": tool.description,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "optional": p.optional,
                        "default": p.default,
                    }
                    for p in tool.parameters
                ],
            }
            for tool in tools
        ],
    }

    lines = ["Bundles:"]
    for name, info in result["bundle_summary"].items():
        lines.append(f"- {name}: {info['count']} tools")

    lines.append("Tools:")
    for tool in tools:
        lines.append(f"- {tool.name} ({tool.category.value}, {tool.cost.value})")

    return result, "\n".join(lines)


def _handle_run_command(args: argparse.Namespace) -> Tuple[Any, Optional[str]]:
    from ts_agents.tools.registry import ToolRegistry
    from ts_agents.tools.executor import ExecutionContext, execute_tool

    try:
        metadata = ToolRegistry.get(args.tool)
    except KeyError:
        tool_names = [tool.name for tool in ToolRegistry.list_all()]
        suggestions = _suggest_tool_names(args.tool, tool_names)
        if suggestions:
            hint = f" Did you mean: {', '.join(suggestions)}?"
        else:
            hint = ""
        raise ValueError(
            f"Tool '{args.tool}' not found.{hint}\n"
            f"Tip: list tools with:\n"
            f"  uv run ts-agents tool list\n"
            f"  uv run ts-agents tool list --json"
        )

    param_types = {param.name: param.type for param in metadata.parameters}
    required = [param.name for param in metadata.parameters if not param.optional]

    params = _parse_param_entries(args.param, param_types)
    params = _apply_run_var_args(params, param_types, args.run_id, args.variable)
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
    )
    execution = execute_tool(args.tool, params, context=context)
    if not execution.success:
        if execution.error:
            raise execution.error
        raise RuntimeError(execution.formatted_output or "Tool execution failed")

    if getattr(args, "json", False):
        return execution.to_dict(), None
    return execution.result, execution.formatted_output or None


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


def _handle_skills_command(args: argparse.Namespace) -> Tuple[Any, str]:
    from pathlib import Path
    from .skills import (
        export_skills,
        validate_all_skills,
        list_skills,
        get_canonical_skills_dir,
        parse_skill_frontmatter,
    )

    if args.skills_command == "export":
        output_path = export_skills(
            args.out,
            agent=getattr(args, "agent", None),
            all_agents=getattr(args, "all_agents", False),
            use_symlinks=getattr(args, "symlink", False),
        )
        result = {"skills_path": str(output_path)}
        if getattr(args, "all_agents", False):
            mode = "symlinks" if getattr(args, "symlink", False) else "copies"
            return result, f"Exported skills as {mode} to all agent directories under {output_path}"
        elif getattr(args, "agent", None):
            mode = "symlinks" if getattr(args, "symlink", False) else "copies"
            return result, f"Exported skills as {mode} to {output_path}"
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

    else:
        raise ValueError(f"Unknown skills command: {args.skills_command}")


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


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
        f"comparison_json: {json.dumps(comparison_payload)}"
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
    import subprocess

    from ts_agents.cli.output import render_output, to_jsonable, write_output
    from ts_agents.core.windowing import (
        select_window_size_from_csv,
        evaluate_windowed_classifier_from_csv,
    )

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _generate_demo_data(args, csv_path)

    value_columns = _split_csv(args.value_columns)
    window_sizes = [int(v) for v in _split_csv(args.window_sizes)] if args.window_sizes else None
    classifier = args.classifier if args.classifier != "auto" else "minirocket"

    selection = select_window_size_from_csv(
        str(csv_path),
        value_columns=value_columns,
        label_column=args.label_column,
        window_sizes=window_sizes,
        metric=args.metric,
        classifier=classifier,
        balance=args.balance,
        max_windows_per_segment=args.max_windows_per_segment,
        test_size=args.test_size,
        seed=args.seed,
    )

    eval_result = evaluate_windowed_classifier_from_csv(
        str(csv_path),
        value_columns=value_columns,
        label_column=args.label_column,
        window_size=int(selection.best_window_size),
        metric=args.metric,
        classifier=classifier,
        balance=args.balance,
        max_windows_per_segment=args.max_windows_per_segment,
        test_size=args.test_size,
        seed=args.seed,
    )

    selection_payload = to_jsonable(selection)
    eval_payload = to_jsonable(eval_result)

    selection_path = output_dir / "window_selection.json"
    eval_path = output_dir / "eval.json"
    write_output(render_output(selection_payload, json_output=True), str(selection_path))
    write_output(render_output(eval_payload, json_output=True), str(eval_path))

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
        "best_window_size": int(selection.best_window_size),
        "metric": str(eval_result.metric),
        "score": float(eval_result.score),
        "classifier": classifier,
        "window_selection_path": str(selection_path),
        "eval_path": str(eval_path),
        "window_scores_plot": str(window_plot) if window_plot.exists() else None,
        "confusion_matrix_plot": str(confusion_plot) if confusion_plot.exists() else None,
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
    selection_path.write_text(json.dumps(selection_payload, indent=2))
    eval_path.write_text(json.dumps(eval_payload, indent=2))

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
    from ts_agents.cli.output import write_output

    artifacts = _run_forecasting_demo_comparison(args)
    report = _build_demo_forecasting_report(
        run_id=args.run_id,
        variable_name=args.variable,
        horizon=args.horizon,
        methods=artifacts["methods"],
        comparison_payload=artifacts["comparison_payload"],
    )
    report_path = _resolve_demo_report_path(
        args,
        default_report_path="outputs/demo/forecasting_report.md",
    )
    write_output(report, str(report_path))

    return {
        "run_id": args.run_id,
        "variable": args.variable,
        "horizon": int(args.horizon),
        "methods": artifacts["methods"],
        "output_dir": str(artifacts["output_dir"]),
        "comparison_path": str(artifacts["comparison_path"]),
        "forecast_comparison_plot": str(artifacts["plot_path"]) if artifacts["plot_path"].exists() else None,
        "best_method": artifacts["best_method"],
        "report_path": str(report_path),
        "report": report,
        "mode": "scripted",
    }


def _run_demo_forecasting_llm(args: argparse.Namespace) -> Dict[str, Any]:
    from ts_agents.cli.output import write_output

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is required for the LLM demo. "
            "Set OPENAI_API_KEY or run with --no-llm (scripted)."
        )

    artifacts = _run_forecasting_demo_comparison(args)
    report = _render_forecasting_report_with_llm(
        model_name=args.model,
        run_id=args.run_id,
        variable_name=args.variable,
        horizon=args.horizon,
        methods=artifacts["methods"],
        comparison_payload=artifacts["comparison_payload"],
    )
    report_path = _resolve_demo_report_path(
        args,
        default_report_path="outputs/demo/forecasting_report.md",
    )
    write_output(report, str(report_path))

    return {
        "run_id": args.run_id,
        "variable": args.variable,
        "horizon": int(args.horizon),
        "methods": artifacts["methods"],
        "output_dir": str(artifacts["output_dir"]),
        "comparison_path": str(artifacts["comparison_path"]),
        "forecast_comparison_plot": str(artifacts["plot_path"]) if artifacts["plot_path"].exists() else None,
        "best_method": artifacts["best_method"],
        "report_path": str(report_path),
        "report": report,
        "mode": "llm",
    }


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


def run(argv: Optional[List[str]] = None) -> int:
    from ts_agents.config import load_user_env

    load_user_env()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if getattr(args, "extract_images", None) and not getattr(args, "save", None):
            raise ValueError("--extract-images requires --save")

        if args.command == "data":
            result, text = _handle_data_command(args)
        elif args.command == "tool":
            result, text = _handle_tool_command(args)
        elif args.command == "run":
            result, text = _handle_run_command(args)
        elif args.command == "agent":
            result, text = _handle_agent_command(args)
        elif args.command == "skills":
            result, text = _handle_skills_command(args)
        elif args.command == "demo":
            result, text = _handle_demo_command(args)
        else:
            parser.error("Unknown command")
            return 2

        output = render_output(result, json_output=args.json, text_output=text)
        print(output)

        if args.save:
            content_to_save = output
            extracted_paths: List[str] = []

            if args.extract_images:
                filename_prefix = Path(args.save).stem or "image"
                if args.json:
                    payload = to_jsonable(result)
                    payload, image_paths = extract_images_from_jsonable(
                        payload,
                        image_dir=args.extract_images,
                        filename_prefix=filename_prefix,
                    )
                    content_to_save = json.dumps(payload, indent=2, default=str)
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
    except Exception as exc:  # pragma: no cover - fall through for CLI errors
        if getattr(args, "json", False):
            error_output = render_output({"error": str(exc)}, json_output=True)
            print(error_output)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
