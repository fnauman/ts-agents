"""Forecast workflow for arbitrary time-series inputs."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional
import warnings

from ts_agents.cli.input_parsing import SeriesInput
from ts_agents.cli.output import dump_json, to_jsonable
from ts_agents.contracts import ToolPayload

from .common import ensure_output_dir, write_dataframe_artifact, write_json_artifact, write_plot_artifact, write_text_artifact

_SUPPORTED_METHODS = {"arima", "ets", "theta"}


def run_forecast_series_workflow(
    series_input: SeriesInput,
    *,
    output_dir: str,
    horizon: int,
    methods: Optional[Iterable[str]] = None,
    validation_size: Optional[int] = None,
    skip_plots: bool = False,
    report_mode: str = "scripted",
    model_name: Optional[str] = None,
) -> ToolPayload:
    """Run the baseline forecasting workflow."""
    import pandas as pd

    from ts_agents.core.comparison import compare_forecasting_methods, plot_forecast_comparison

    workflow_name = "forecast-series"
    output_path = ensure_output_dir(output_dir)
    selected_methods = _normalize_methods(methods)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comparison = compare_forecasting_methods(
            series_input.series,
            horizon=horizon,
            methods=selected_methods,
            validation_size=validation_size,
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
    )
    warnings_list: List[str] = []
    quality_flags = _forecast_quality_flags(
        selected_methods=selected_methods,
        valid_methods=valid_methods,
        failed_methods=failed_methods,
    )
    if not rmse_ranking:
        quality_flags.append("ranking_unavailable")
    if failed_methods:
        warnings_list.append(
            "Some forecast methods failed: "
            + ", ".join(
                f"{method} ({(comparison_payload.get('metrics') or {}).get(method, {}).get('error', 'unknown error')})"
                for method in failed_methods
            )
        )

    summary_data = {
        "workflow": workflow_name,
        "source": series_input.provenance.get("series_ref", {}),
        "horizon": int(horizon),
        "validation_size": int(validation_size or horizon),
        "methods": selected_methods,
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
    return ToolPayload(
        kind="workflow",
        summary=summary,
        status="degraded" if warnings_list or quality_flags else "ok",
        data=summary_data,
        artifacts=artifacts,
        warnings=warnings_list,
        provenance=series_input.provenance,
    )


def _normalize_methods(methods: Optional[Iterable[str]]) -> List[str]:
    if methods is None:
        return ["arima", "theta"]

    normalized = [str(method).strip().lower() for method in methods if str(method).strip()]
    if not normalized:
        return ["arima", "theta"]

    invalid = [method for method in normalized if method not in _SUPPORTED_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported forecasting methods: {', '.join(invalid)}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_METHODS))}."
        )
    return normalized


def _failed_methods(
    selected_methods: List[str],
    comparison_payload: dict[str, Any],
) -> List[str]:
    metrics = comparison_payload.get("metrics") or {}
    return [
        method
        for method in selected_methods
        if isinstance(metrics.get(method), dict) and "error" in metrics[method]
    ]


def _valid_methods(
    selected_methods: List[str],
    comparison_payload: dict[str, Any],
) -> List[str]:
    metrics = comparison_payload.get("metrics") or {}
    return [
        method
        for method in selected_methods
        if isinstance(metrics.get(method), dict) and "error" not in metrics[method]
    ]


def _forecast_quality_flags(
    *,
    selected_methods: List[str],
    valid_methods: List[str],
    failed_methods: List[str],
) -> List[str]:
    flags: List[str] = []
    if failed_methods:
        flags.append("partial_method_failure")
    if len(valid_methods) == 1 and len(selected_methods) > 1:
        flags.append("only_one_valid_method")
    return flags


def _raise_all_methods_failed(
    selected_methods: List[str],
    comparison_payload: dict[str, Any],
) -> None:
    metrics = comparison_payload.get("metrics") or {}
    failure_messages = {
        method: metrics.get(method, {}).get("error", "unknown error")
        for method in selected_methods
    }
    message = (
        "All forecast methods failed: "
        + "; ".join(f"{method}: {error}" for method, error in failure_messages.items())
    )
    if all(_looks_like_dependency_failure(error) for error in failure_messages.values()):
        raise ImportError(message)
    raise RuntimeError(message)


def _looks_like_dependency_failure(message: Any) -> bool:
    if not isinstance(message, str):
        return False
    lowered = message.lower()
    return "optional dependencies" in lowered or "install with" in lowered


def _build_forecast_rows(
    *,
    series_input: SeriesInput,
    method: Optional[str],
    horizon: int,
) -> List[dict[str, Any]]:
    if method is None:
        return []

    result = _forecast_with_method(series_input.series, method=method, horizon=horizon)
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


def _forecast_with_method(series, *, method: str, horizon: int):
    from ts_agents.core.forecasting import forecast_arima, forecast_ets, forecast_theta

    method_map = {
        "arima": forecast_arima,
        "ets": forecast_ets,
        "theta": forecast_theta,
    }
    return method_map[method](series, horizon=horizon)


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
