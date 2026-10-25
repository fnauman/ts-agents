"""Inspection workflow for arbitrary time-series inputs."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ts_agents.cli.input_parsing import SeriesInput
from ts_agents.cli.output import to_jsonable
from ts_agents.contracts import AnalysisLedgerEntry, ForecastRecommendation, ToolPayload

from .common import (
    attach_workflow_run_metadata,
    ensure_output_dir,
    write_json_artifact,
    write_ledger_artifact,
    write_plot_artifact,
    write_text_artifact,
)


def run_inspect_series_workflow(
    series_input: SeriesInput,
    *,
    output_dir: str,
    max_lag: Optional[int] = None,
    skip_plots: bool = False,
    run_id: Optional[str] = None,
    resumed: bool = False,
    output_dir_mode: str = "explicit",
    defer_finalization: bool = False,
) -> ToolPayload:
    """Run task-neutral diagnostics on a series and suggest next steps.

    Reports data quality (sampling, gaps, reordering), descriptive statistics,
    significance-tested periodicity, autocorrelation, a quick regime screen
    (when the patterns extra is installed), and next steps across task
    families. A forecasting start method is included as one section.
    """
    from ts_agents.core.recommendations import recommend_forecasting_method
    from ts_agents.core.spectral import detect_periodicity
    from ts_agents.core.spectral.psd import MIN_PERIODICITY_SAMPLES
    from ts_agents.core.statistics import compute_autocorrelation, describe_series

    workflow_name = "inspect-series"
    output_path = ensure_output_dir(output_dir)
    series = series_input.series
    n = len(series)
    time_axis = series_input.time_axis
    warnings: List[str] = list((time_axis or {}).get("warnings") or [])
    quality_flags: List[str] = []
    if time_axis and time_axis.get("regular") is False:
        quality_flags.append("irregular_sampling")
    if time_axis and time_axis.get("reordered"):
        quality_flags.append("input_reordered")

    stats = describe_series(series, extended=True)

    periodicity = None
    if n >= MIN_PERIODICITY_SAMPLES and stats.std > 0:
        periodicity = detect_periodicity(series, top_n=3)
    else:
        reason = "constant" if stats.std == 0 else f"shorter than {MIN_PERIODICITY_SAMPLES} samples"
        warnings.append(f"Periodicity not assessed: series is {reason}.")
        quality_flags.append("periodicity_not_assessed")

    resolved_max_lag = max_lag if max_lag is not None else min(64, max(8, n // 4))
    acf = np.array([1.0])
    if stats.std > 0 and n >= 2:
        acf = compute_autocorrelation(series, max_lag=resolved_max_lag)
    regime_screen = _regime_screen(series)
    forecast_recommendation = recommend_forecasting_method(
        stats=stats,
        periodicity=periodicity,
        acf=acf,
        available_methods=_available_forecasting_methods(),
    )

    recommended_next_steps = _recommend_next_steps(
        periodicity=periodicity,
        acf=acf,
        length=n,
        regime_screen=regime_screen,
        time_axis=time_axis,
    )
    effective_max_lag = max(len(acf) - 1, 0)
    lag_one = float(acf[1]) if len(acf) > 1 else None
    periodicity_payload = to_jsonable(periodicity) if periodicity is not None else {
        "detected": False,
        "dominant_period": None,
        "assessed": False,
    }
    if periodicity is not None:
        periodicity_payload["assessed"] = True
        periodicity_payload["period_unit"] = "samples"
        sample_rate = (time_axis or {}).get("sample_rate")
        if sample_rate and periodicity.dominant_period:
            periodicity_payload["dominant_period_time"] = periodicity.dominant_period / sample_rate
            periodicity_payload["time_unit"] = (time_axis or {}).get("step_unit")
    ledger_entries = _build_analysis_ledger(
        stats=to_jsonable(stats),
        periodicity=periodicity_payload,
        lag_one=lag_one,
        forecast_recommendation=forecast_recommendation,
    )
    summary_data = {
        "workflow": workflow_name,
        "source": series_input.provenance.get("series_ref", {}),
        "series_length": int(n),
        "time_axis": time_axis,
        "stats": to_jsonable(stats),
        "periodicity": periodicity_payload,
        "autocorrelation": {
            "max_lag": int(effective_max_lag),
            "requested_max_lag": int(resolved_max_lag),
            "lag_1": lag_one,
        },
        "regime_screen": regime_screen,
        "quality_flags": quality_flags,
        "recommended_next_steps": recommended_next_steps,
        "forecast_recommendation": to_jsonable(forecast_recommendation),
        "output_dir": str(output_path),
    }

    artifacts = [
        write_json_artifact(
            data=summary_data,
            path=output_path / "summary.json",
            description="Inspection summary JSON.",
            created_by=workflow_name,
        ),
        write_ledger_artifact(
            entries=ledger_entries,
            path=output_path / "ledger.json",
            created_by=workflow_name,
        ),
    ]

    if not skip_plots:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(np.arange(len(acf)), acf)
            ax.set_title("Autocorrelation")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")
            fig.tight_layout()
            artifacts.append(
                write_plot_artifact(
                    figure=fig,
                    path=output_path / "autocorrelation.png",
                    description="Autocorrelation plot generated by the inspect-series workflow.",
                    created_by=workflow_name,
                )
            )
            plt.close(fig)
        except ImportError:
            warnings.append("matplotlib is not installed; skipping autocorrelation plot.")

    report = _build_report(
        series_input=series_input,
        summary_data=summary_data,
        warnings=warnings,
    )
    artifacts.append(
        write_text_artifact(
            content=report,
            path=output_path / "report.md",
            description="Inspection workflow markdown report.",
            created_by=workflow_name,
        )
    )

    dominant_period = summary_data["periodicity"].get("dominant_period")
    period_text = (
        f"significant period {_format_number(dominant_period)} samples"
        if dominant_period is not None
        else "no significant periodicity"
    )
    regime_text = (
        f"{len(regime_screen['changepoints'])} candidate regime change(s)"
        if regime_screen.get("ran")
        else "regime screen skipped"
    )
    summary = (
        f"Inspect-series workflow completed for {series_input.label}: {period_text}; "
        f"lag-1 autocorrelation {_format_number(lag_one)}; {regime_text}."
    )
    payload = ToolPayload(
        kind="workflow",
        summary=summary,
        status="degraded" if warnings else "ok",
        data=summary_data,
        artifacts=artifacts,
        warnings=warnings,
        provenance=series_input.provenance,
    )
    return attach_workflow_run_metadata(
        payload,
        workflow_name=workflow_name,
        output_dir=output_path,
        run_id=run_id,
        source=summary_data["source"],
        options={
            "max_lag": max_lag,
            "skip_plots": skip_plots,
        },
        resumed=resumed,
        output_dir_mode=output_dir_mode,
        defer_finalization=defer_finalization,
    )


def _available_forecasting_methods() -> list[str]:
    from ts_agents.workflows import get_workflow

    availability = get_workflow("forecast-series").availability()
    methods = availability.get("available_methods") or ["seasonal_naive"]
    normalized = [str(method) for method in methods if str(method)]
    return normalized or ["seasonal_naive"]


def _regime_screen(series: np.ndarray) -> dict[str, Any]:
    """Quick scale-invariant change-point screen; skipped without the patterns extra."""
    try:
        from ts_agents.core.patterns import segment_changepoint

        result = segment_changepoint(series)
    except ImportError:
        return {"ran": False, "reason": "ruptures is not installed (ts-agents[patterns])."}
    except ValueError as exc:
        return {"ran": False, "reason": str(exc)}
    return {
        "ran": True,
        "method": result.method,
        "changepoints": [int(cp) for cp in result.changepoints],
        "settings": to_jsonable(result.settings),
        "warnings": list(result.warnings),
    }


def _recommend_next_steps(
    *,
    periodicity: Any,
    acf: np.ndarray,
    length: int,
    regime_screen: Optional[dict[str, Any]] = None,
    time_axis: Optional[dict[str, Any]] = None,
) -> List[str]:
    steps: List[str] = []
    if time_axis and time_axis.get("regular") is False:
        steps.append(
            "Resolve irregular sampling or gaps before spectral, decomposition, or forecasting steps; "
            "they assume equally spaced samples."
        )
    period = getattr(periodicity, "dominant_period", None) if periodicity is not None else None
    if period is not None:
        rounded = max(2, int(round(period)))
        steps.append(
            f"Decompose with stl_decompose --param period={rounded} to separate trend, seasonality, and residual."
        )
        steps.append(
            f"For anomalies, run find_discords with m near the period ({rounded} samples) and judge results by robust_z."
        )
        steps.append(
            f"For forecasting, run forecast-series with --season-length {rounded} and a horizon of at least one period."
        )
    else:
        steps.append(
            "No significant periodicity: use find_discords with an event-length m for anomalies, and "
            "non-seasonal baselines if forecasting."
        )
    if regime_screen and regime_screen.get("ran") and regime_screen.get("changepoints"):
        steps.append(
            "Candidate regime changes at samples "
            f"{regime_screen['changepoints'][:10]}: analyze segments separately (segment_changepoint) "
            "before fitting one model to the whole series."
        )
    elif regime_screen and not regime_screen.get("ran"):
        steps.append(f"Regime screen skipped ({regime_screen.get('reason')}).")
    if len(acf) > 1 and abs(float(acf[1])) >= 0.5:
        steps.append("Strong short-lag dependence: residual-based anomaly checks should use decomposition residuals.")
    if length < 64:
        steps.append("The series is short; treat all diagnostics as low-confidence.")
    return steps


def _build_analysis_ledger(
    *,
    stats: dict[str, Any],
    periodicity: dict[str, Any],
    lag_one: float | None,
    forecast_recommendation: ForecastRecommendation,
) -> List[AnalysisLedgerEntry]:
    entries: List[AnalysisLedgerEntry] = [
        AnalysisLedgerEntry(
            kind="finding",
            key="series_length",
            value=int(stats.get("length") or 0),
            confidence=1.0,
            evidence=["summary.json#stats"],
        ),
        AnalysisLedgerEntry(
            kind="finding",
            key="dominant_period",
            value=periodicity.get("dominant_period"),
            confidence=_optional_float(periodicity.get("confidence")),
            evidence=["summary.json#periodicity"],
            notes="Period that passed the red-noise significance test (null when none did).",
        ),
        AnalysisLedgerEntry(
            kind="finding",
            key="lag_1_autocorrelation",
            value=lag_one,
            confidence=min(abs(lag_one), 1.0) if lag_one is not None else None,
            evidence=["summary.json#autocorrelation"],
        ),
        AnalysisLedgerEntry(
            kind="recommendation",
            key="forecasting_method",
            value=forecast_recommendation.choice,
            confidence=forecast_recommendation.confidence,
            evidence=[
                "summary.json#stats",
                "summary.json#periodicity",
                "summary.json#autocorrelation",
            ],
            notes="; ".join(forecast_recommendation.rationale),
        ),
    ]
    return entries


def _build_report(
    *,
    series_input: SeriesInput,
    summary_data: dict[str, Any],
    warnings: List[str],
) -> str:
    stats = summary_data["stats"]
    periodicity = summary_data["periodicity"]
    forecast_recommendation = summary_data["forecast_recommendation"]
    regime = summary_data["regime_screen"]
    time_axis = summary_data.get("time_axis") or {}
    top_periods = periodicity.get("top_periods") or []
    rationale = forecast_recommendation.get("rationale") or []
    if regime.get("ran"):
        regime_text = f"{regime.get('changepoints')} ({regime.get('method')})"
    else:
        regime_text = f"skipped: {regime.get('reason')}"
    if time_axis:
        sampling_text = (
            f"{'regular' if time_axis.get('regular') else 'irregular'}; median step "
            f"{_format_number(time_axis.get('median_step'))} {time_axis.get('step_unit', '')}; "
            f"gaps {time_axis.get('gap_count', 0)}"
        )
    else:
        sampling_text = "no time column (sample index)"
    return "\n".join(
        [
            "### Report on Inspect-Series Workflow",
            "",
            "#### Data",
            f"- **Source**: `{series_input.label}`",
            f"- **Length**: {stats.get('length')} (non-finite dropped: {stats.get('nan_count', 0)})",
            f"- **Sampling**: {sampling_text}",
            f"- **Mean / Std**: {_format_number(stats.get('mean'))} / {_format_number(stats.get('std'))}",
            f"- **Min / Max**: {_format_number(stats.get('min'))} / {_format_number(stats.get('max'))}",
            f"- **Quality Flags**: {', '.join(summary_data.get('quality_flags') or []) or 'none'}",
            f"- **Warnings**: {'; '.join(warnings) if warnings else 'none'}",
            "",
            "#### Structure",
            f"- **Significant Period (samples)**: {_format_number(periodicity.get('dominant_period'))}",
            f"- **Periodicity p-value**: {_format_number(periodicity.get('p_value'))}",
            f"- **Strongest Candidate (samples)**: {_format_number(periodicity.get('candidate_period'))}",
            f"- **Significant Periods**: {', '.join(_format_number(value) for value in top_periods) if top_periods else 'none'}",
            f"- **Lag-1 Autocorrelation**: {_format_number(summary_data['autocorrelation'].get('lag_1'))}",
            f"- **Regime Screen**: {regime_text}",
            "",
            "#### Recommended Next Steps",
            *[f"- {step}" for step in summary_data["recommended_next_steps"]],
            "",
            "#### If Forecasting",
            f"- **Start Method**: {forecast_recommendation.get('choice', 'n/a')}",
            f"- **Recommendation Confidence**: {_format_number(forecast_recommendation.get('confidence'))}",
            *([f"- {item}" for item in rationale] or ["- n/a"]),
        ]
    )


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
