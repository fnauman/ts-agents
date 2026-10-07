"""Forecast workflow for arbitrary time-series inputs."""

from __future__ import annotations

from contextlib import ExitStack
import numbers
from typing import Any, Iterable, List, Optional
import warnings

from ts_agents.cli.input_parsing import SeriesInput
from ts_agents.cli.output import dump_json, to_jsonable
from ts_agents.contracts import ToolPayload
from ts_agents.core.forecasting.catalog import foundation_methods, methods_for

from .common import (
    attach_workflow_run_metadata,
    ensure_output_dir,
    write_dataframe_artifact,
    write_json_artifact,
    write_plot_artifact,
    write_text_artifact,
)

_SUPPORTED_METHODS = set(methods_for("series"))
_FOUNDATION_METHODS = frozenset(foundation_methods("series"))


def run_forecast_series_workflow(
    series_input: SeriesInput,
    *,
    output_dir: str,
    horizon: int,
    methods: Optional[Iterable[str]] = None,
    season_length: Optional[int] = None,
    validation_size: Optional[int] = None,
    skip_plots: bool = False,
    report_mode: str = "scripted",
    model_name: Optional[str] = None,
    run_id: Optional[str] = None,
    resumed: bool = False,
    output_dir_mode: str = "explicit",
    defer_finalization: bool = False,
    context_length: Optional[int] = None,
    accelerator: Optional[str] = None,
) -> ToolPayload:
    """Run the baseline forecasting workflow.

    Foundation-model methods (for example ``chronos2_small``) run zero-shot and use
    ``context_length`` and ``accelerator``; statistical methods ignore both.
    """
    workflow_name = "forecast-series"
    selected_methods = _normalize_methods(methods)
    foundation_requested = [method for method in selected_methods if method in _FOUNDATION_METHODS]
    # Validated whatever the methods (as forecast-panel does): it enters the resume identity.
    if context_length is not None and (
        isinstance(context_length, bool)
        or not isinstance(context_length, numbers.Integral)
        or context_length < 1
    ):
        raise ValueError("context_length must be a positive integer.")
    if foundation_requested:
        _validate_foundation_requests(
            foundation_requested,
            horizons={int(validation_size or horizon), int(horizon)},
            context_length=context_length,
            accelerator=accelerator,
        )
    # Only user-set options are forwarded, so statistical-only calls stay unchanged.
    foundation_options = {
        key: value
        for key, value in (("context_length", context_length), ("accelerator", accelerator))
        if value is not None
    }
    output_path = ensure_output_dir(output_dir)

    model_scope = ExitStack()
    if foundation_requested:
        from ts_agents.core.forecasting.foundation import model_cache_scope

        # Models stay loaded for the validation and final forecasts, then are released.
        model_scope.enter_context(model_cache_scope())
    try:
        return _run_forecast_series(
            series_input,
            workflow_name=workflow_name,
            output_path=output_path,
            horizon=horizon,
            selected_methods=selected_methods,
            foundation_requested=foundation_requested,
            foundation_options=foundation_options,
            season_length=season_length,
            validation_size=validation_size,
            skip_plots=skip_plots,
            report_mode=report_mode,
            model_name=model_name,
            run_id=run_id,
            resumed=resumed,
            output_dir_mode=output_dir_mode,
            defer_finalization=defer_finalization,
        )
    finally:
        model_scope.close()


def _run_forecast_series(
    series_input: SeriesInput,
    *,
    workflow_name: str,
    output_path,
    horizon: int,
    selected_methods: List[str],
    foundation_requested: List[str],
    foundation_options: dict[str, Any],
    season_length: Optional[int],
    validation_size: Optional[int],
    skip_plots: bool,
    report_mode: str,
    model_name: Optional[str],
    run_id: Optional[str],
    resumed: bool,
    output_dir_mode: str,
    defer_finalization: bool,
) -> ToolPayload:
    import pandas as pd

    from ts_agents.core.comparison import compare_forecasting_methods, plot_forecast_comparison

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comparison = compare_forecasting_methods(
            series_input.series,
            horizon=horizon,
            methods=selected_methods,
            season_length=season_length,
            validation_size=validation_size,
            **foundation_options,
        )

    comparison_payload = to_jsonable(comparison)
    valid_methods = _valid_methods(selected_methods, comparison_payload)
    failed_methods = _failed_methods(selected_methods, comparison_payload)
    if not valid_methods:
        _raise_all_methods_failed(selected_methods, comparison_payload)

    rankings = comparison_payload.get("rankings") or {}
    rmse_ranking = rankings.get("rmse") or []
    best_method = rmse_ranking[0] if rmse_ranking else None
    if best_method is None:
        best_method = valid_methods[0]

    forecast_rows = _build_forecast_rows(
        series_input=series_input,
        method=best_method,
        horizon=horizon,
        season_length=season_length,
        **foundation_options,
    )
    warnings_list: List[str] = []
    quality_flags = _forecast_quality_flags(
        selected_methods=selected_methods,
        valid_methods=valid_methods,
        failed_methods=failed_methods,
        comparison_payload=comparison_payload,
    )
    if not rmse_ranking:
        quality_flags.append("ranking_unavailable")
    if failed_methods:
        warnings_list.append(
            "Some forecast methods failed: "
            + ", ".join(
                f"{method} ({_method_failure_message(method, comparison_payload)})"
                for method in failed_methods
            )
        )

    summary_data = {
        "workflow": workflow_name,
        "source": series_input.provenance.get("series_ref", {}),
        "horizon": int(horizon),
        "validation_size": int(validation_size or horizon),
        "methods": selected_methods,
        "season_length": int(season_length) if season_length else None,
        "best_method": best_method,
        "valid_methods": valid_methods,
        "failed_methods": failed_methods,
        "quality_flags": quality_flags,
        "metrics": comparison_payload.get("metrics", {}),
        "rankings": comparison_payload.get("rankings", {}),
        "recommendation": comparison_payload.get("recommendation"),
        "forecast": forecast_rows,
        "output_dir": str(output_path),
    }
    # Provenance and the zero-shot note cover only foundation models that produced metrics.
    foundation_ran = [method for method in foundation_requested if method in valid_methods]
    foundation_failed = [method for method in foundation_requested if method in failed_methods]
    foundation_specs: dict[str, dict[str, Any]] = {}
    if foundation_ran:
        from ts_agents.core.forecasting.foundation import model_spec

        context_length = foundation_options.get("context_length")
        accelerator = foundation_options.get("accelerator")
        # Validation and final forecasts both run at their own horizon; record the final one.
        foundation_specs = {
            method: model_spec(
                method,
                horizon=horizon,
                context_length=context_length,
                accelerator=accelerator,
            )
            for method in foundation_ran
        }
        summary_data["foundation_models"] = foundation_specs

    artifacts = [
        write_json_artifact(
            data=comparison_payload,
            path=output_path / "forecast_comparison.json",
            description="Forecast comparison metrics and rankings.",
            created_by=workflow_name,
        ),
        write_json_artifact(
            data=forecast_rows,
            path=output_path / "forecast.json",
            description="Best-model forecast in JSON form.",
            created_by=workflow_name,
        ),
        write_dataframe_artifact(
            dataframe=pd.DataFrame(forecast_rows),
            path=output_path / "forecast.csv",
            description="Best-model forecast as CSV.",
            created_by=workflow_name,
        ),
    ]
    for method, spec in foundation_specs.items():
        artifacts.append(
            write_json_artifact(
                data=spec,
                path=output_path / "models" / method / "model_spec.json",
                description=f"{method} checkpoint provenance (zero-shot; no weights saved).",
                created_by=workflow_name,
            )
        )
    if not skip_plots:
        try:
            fig = plot_forecast_comparison(comparison, series_input.series)
            artifacts.append(
                write_plot_artifact(
                    figure=fig,
                    path=output_path / "forecast_comparison.png",
                    description="Forecast comparison plot.",
                    created_by=workflow_name,
                )
            )
            import matplotlib.pyplot as plt

            plt.close(fig)
        except ImportError:
            warnings_list.append("matplotlib is not installed; skipping forecast comparison plot.")
            quality_flags.append("plot_skipped")

    report = _build_report(
        series_input=series_input,
        horizon=horizon,
        methods=selected_methods,
        season_length=season_length,
        comparison_payload=comparison_payload,
    )
    if report_mode == "llm":
        report = _render_report_with_llm(
            model_name=model_name,
            source_label=series_input.label,
            horizon=horizon,
            methods=selected_methods,
            comparison_payload=comparison_payload,
        )
    if foundation_requested:
        report += "\n\n" + _foundation_report_section(foundation_specs, foundation_failed)
    artifacts.append(
        write_text_artifact(
            content=report,
            path=output_path / "report.md",
            description="Forecast workflow markdown report.",
            created_by=workflow_name,
        )
    )

    best_method_text = best_method or "n/a"
    summary = (
        f"Forecast-series workflow completed for {series_input.label} "
        f"with horizon {horizon}. Best method by RMSE: {best_method_text}."
    )
    payload = ToolPayload(
        kind="workflow",
        summary=summary,
        status="degraded" if warnings_list or quality_flags else "ok",
        data=summary_data,
        artifacts=artifacts,
        warnings=warnings_list,
        provenance=series_input.provenance,
    )
    return attach_workflow_run_metadata(
        payload,
        workflow_name=workflow_name,
        output_dir=output_path,
        run_id=run_id,
        source=summary_data["source"],
        options={
            "horizon": horizon,
            "methods": selected_methods,
            "season_length": season_length,
            "validation_size": validation_size,
            "skip_plots": skip_plots,
            "report_mode": report_mode,
            **foundation_options,
        },
        resumed=resumed,
        output_dir_mode=output_dir_mode,
        defer_finalization=defer_finalization,
    )


def _normalize_methods(methods: Optional[Iterable[str]]) -> List[str]:
    if methods is None:
        return ["seasonal_naive", "arima", "theta"]

    normalized = [str(method).strip().lower() for method in methods if str(method).strip()]
    if not normalized:
        return ["seasonal_naive", "arima", "theta"]

    invalid = [method for method in normalized if method not in _SUPPORTED_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported forecasting methods: {', '.join(invalid)}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_METHODS))}."
        )
    return normalized


def _validate_foundation_requests(
    methods: List[str],
    *,
    horizons: set[int],
    context_length: Optional[int],
    accelerator: Optional[str],
) -> None:
    """Reject horizon/context/device requests a checkpoint cannot serve, before any work."""
    from ts_agents.core.forecasting.foundation import ACCELERATORS, validate_request

    if accelerator is not None and accelerator not in ACCELERATORS:
        raise ValueError(f"accelerator must be one of {', '.join(ACCELERATORS)}, got {accelerator!r}.")
    for method in methods:
        for value in sorted(horizons):
            validate_request(method, horizon=value, context_length=context_length)


def _failed_methods(
    selected_methods: List[str],
    comparison_payload: dict[str, Any],
) -> List[str]:
    return [
        method
        for method in selected_methods
        if _method_failure_message(method, comparison_payload) is not None
    ]


def _valid_methods(
    selected_methods: List[str],
    comparison_payload: dict[str, Any],
) -> List[str]:
    return [
        method
        for method in selected_methods
        if _method_failure_message(method, comparison_payload) is None
    ]


def _method_metric_entry(
    method: str,
    comparison_payload: dict[str, Any],
) -> Any:
    metrics = comparison_payload.get("metrics") or {}
    return metrics.get(method)


def _method_failure_message(
    method: str,
    comparison_payload: dict[str, Any],
) -> Optional[str]:
    metric_entry = _method_metric_entry(method, comparison_payload)
    if not isinstance(metric_entry, dict):
        return "missing metrics for method"
    error = metric_entry.get("error")
    if error is None:
        return None
    return str(error)


def _method_failure_type(
    method: str,
    comparison_payload: dict[str, Any],
) -> Optional[str]:
    metric_entry = _method_metric_entry(method, comparison_payload)
    if not isinstance(metric_entry, dict):
        return "MissingMetricsError"
    error = metric_entry.get("error")
    if error is None:
        return None
    error_type = metric_entry.get("error_type")
    return str(error_type) if error_type else None


def _forecast_quality_flags(
    *,
    selected_methods: List[str],
    valid_methods: List[str],
    failed_methods: List[str],
    comparison_payload: dict[str, Any],
) -> List[str]:
    flags: List[str] = []
    if failed_methods:
        flags.append("partial_method_failure")
    if len(valid_methods) == 1 and len(selected_methods) > 1:
        flags.append("only_one_valid_method")
    if any(
        _method_failure_message(method, comparison_payload) == "missing metrics for method"
        for method in failed_methods
    ):
        flags.append("missing_method_metrics")
    return flags


def _foundation_report_section(
    foundation_specs: dict[str, dict[str, Any]],
    foundation_failed: List[str],
) -> str:
    lines = ["#### Foundation models"]
    if foundation_specs:
        lines.append(
            f"{', '.join(foundation_specs)} ran zero-shot through Darts: nothing was trained on "
            "this series, each forecast conditions only on the last context_length points before "
            "its cutoff, and pretraining corpora may overlap public benchmarks."
        )
        for method, spec in foundation_specs.items():
            lines.append(
                f"- {method}: {spec['darts_class']} from "
                f"{spec['hub_model_name']}@{spec['hub_model_revision']}, "
                f"context {spec['input_chunk_length'][1]}, licence {spec['license']}."
            )
        lines.append("models/<foundation model>/ holds model_spec.json only; weights are not saved.")
    if foundation_failed:
        lines.append(
            f"Requested but failed (see warnings; no forecasts produced): {', '.join(foundation_failed)}."
        )
    return "\n".join(lines)


def _raise_all_methods_failed(
    selected_methods: List[str],
    comparison_payload: dict[str, Any],
) -> None:
    failure_messages = {
        method: _method_failure_message(method, comparison_payload) or "unknown error"
        for method in selected_methods
    }
    message = (
        "All forecast methods failed: "
        + "; ".join(f"{method}: {error}" for method, error in failure_messages.items())
    )
    failure_types = {
        method: _method_failure_type(method, comparison_payload)
        for method in selected_methods
    }
    if all(
        _is_dependency_failure(
            error_type=failure_types[method],
            message=failure_messages[method],
        )
        for method in selected_methods
    ):
        raise ImportError(message)
    if all(
        failure_types[method] == "FoundationModelUnavailableError"
        for method in selected_methods
    ):
        from ts_agents.core.forecasting.foundation import FoundationModelUnavailableError

        # Keep the typed weights-unavailable failure (backend_unavailable, exit 5).
        raise FoundationModelUnavailableError(message)
    raise RuntimeError(message)


def _is_dependency_failure(*, error_type: Optional[str], message: Any) -> bool:
    if error_type in {"ImportError", "ModuleNotFoundError"}:
        return True

    if not isinstance(message, str):
        return False

    lowered = message.lower()
    dependency_markers = (
        "optional dependencies",
        "dependency",
        "dependencies",
        "no module named",
        "module not found",
        "modulenotfounderror",
        "importerror",
    )
    install_markers = (
        "pip install",
        "install via",
        "install with",
    )
    return any(marker in lowered for marker in dependency_markers) or (
        any(marker in lowered for marker in install_markers)
        and ("require" in lowered or "missing" in lowered)
    )


def _build_forecast_rows(
    *,
    series_input: SeriesInput,
    method: Optional[str],
    horizon: int,
    season_length: Optional[int] = None,
    **options: Any,
) -> List[dict[str, Any]]:
    if method is None:
        return []

    result = _forecast_with_method(
        series_input.series,
        method=method,
        horizon=horizon,
        season_length=season_length,
        **options,
    )
    forecast_values = to_jsonable(result.forecast)
    future_index = _infer_future_index(series_input=series_input, horizon=horizon)
    rows: List[dict[str, Any]] = []
    for step, forecast_value in enumerate(forecast_values, start=1):
        row = {
            "step": step,
            "forecast": forecast_value,
        }
        if future_index is not None:
            row["time"] = future_index[step - 1]
        rows.append(row)
    return rows


def _forecast_with_method(
    series,
    *,
    method: str,
    horizon: int,
    season_length: Optional[int] = None,
    **options: Any,
):
    from ts_agents.core.forecasting.catalog import get_series_forecaster

    kwargs = dict(options)
    if season_length is not None:
        kwargs["season_length"] = season_length
    # The catalog forecaster drops options a method does not accept.
    return get_series_forecaster(method)(series, horizon=horizon, **kwargs)


def _infer_future_index(
    *,
    series_input: SeriesInput,
    horizon: int,
) -> Optional[List[Any]]:
    if not series_input.time_values:
        return None

    import pandas as pd

    time_index = pd.Index(series_input.time_values)
    if len(time_index) < 2:
        return None

    if not pd.api.types.is_datetime64_any_dtype(time_index.dtype):
        parsed_time = pd.to_datetime(time_index, errors="coerce")
        if not parsed_time.isna().any():
            time_index = pd.DatetimeIndex(parsed_time)

    if pd.api.types.is_datetime64_any_dtype(time_index.dtype):
        datetime_index = pd.DatetimeIndex(time_index)
        inferred_freq = pd.infer_freq(datetime_index)
        if inferred_freq:
            start = datetime_index[-1]
            future = pd.date_range(start=start, periods=horizon + 1, freq=inferred_freq)[1:]
            return [value.isoformat() for value in future]
        return None

    try:
        numeric_values = [float(value) for value in time_index.to_list()]
    except (TypeError, ValueError):
        return None

    step = numeric_values[-1] - numeric_values[-2]
    if step == 0:
        return None
    return [numeric_values[-1] + step * offset for offset in range(1, horizon + 1)]


def _build_report(
    *,
    series_input: SeriesInput,
    horizon: int,
    methods: List[str],
    season_length: Optional[int],
    comparison_payload: dict[str, Any],
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
            f"- `{method}`: RMSE={_format_metric(method_metric.get('rmse'))}, "
            f"MAE={_format_metric(method_metric.get('mae'))}, "
            f"MAPE={_format_metric(method_metric.get('mape'))}%"
        )

    return "\n".join(
        [
            "### Report on Forecast-Series Workflow",
            "",
            f"- **Source**: `{series_input.label}`",
            f"- **Horizon**: {horizon}",
            f"- **Season Length**: {season_length if season_length is not None else 'n/a'}",
            f"- **Compared Methods**: {', '.join(methods)}",
            f"- **Best Method (RMSE)**: {best_method}",
            "",
            "#### Metrics",
            *method_lines,
            "",
            "#### Recommendation",
            recommendation,
        ]
    )


def _render_report_with_llm(
    *,
    model_name: Optional[str],
    source_label: str,
    horizon: int,
    methods: List[str],
    comparison_payload: dict[str, Any],
) -> str:
    from langchain_openai import ChatOpenAI
    from ts_agents.config import get_openai_model

    llm = ChatOpenAI(model=model_name or get_openai_model(), temperature=0)
    prompt = (
        "Write a concise markdown report for a forecasting workflow.\n"
        "Use <= 14 lines and include: source, horizon, compared methods, "
        "best method by RMSE (if available), key metrics, and one caveat.\n\n"
        f"source: {source_label}\n"
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
        return "\n".join(part for part in parts if part)
    return str(content)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)
