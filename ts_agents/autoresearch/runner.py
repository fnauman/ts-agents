"""Execution logic for built-in autoresearch loops."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.util import find_spec
import json
from pathlib import Path
import signal
import stat
import threading
import time
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from ts_agents.cli.input_parsing import load_labeled_stream_input
from ts_agents.cli.output import render_output, to_jsonable, write_output
from ts_agents.contracts import ArtifactRef, CLI_SCHEMA_VERSION
from ts_agents.runtime_paths import resolve_existing_path
from ts_agents.workflows.common import (
    WORKFLOW_MANIFEST_FILENAME,
    artifact_ref,
    clear_output_dir,
    ensure_output_dir,
    generate_workflow_run_id,
    output_dir_has_files,
)

from .registry import get_loop, loop_to_dict


DEFAULT_FORECAST_SERIES = ["M4", "M10", "M100", "M1000", "M1002"]
DEFAULT_FORECAST_METHODS = ["seasonal_naive", "theta", "ets", "arima"]
DEFAULT_CLASSIFIERS = ["knn", "minirocket", "rocket"]
DEFAULT_FOUNDATION_MODELS = ["amazon/chronos-2", "amazon/chronos-t5-small", "AutonLab/MOMENT-1-large"]
DEFAULT_WINDOW_SIZES = [32, 64, 96, 128, 160]
_STATSFORECAST_METHODS = {"theta", "ets", "arima"}


def run_autoresearch_loop(
    *,
    loop_name: str,
    output_dir: str,
    profile: str = "default",
    models: Optional[Iterable[str]] = None,
    max_trials: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    skip_plots: bool = False,
    seed: int = 1337,
    dataset: str = "auto",
) -> dict[str, Any]:
    """Run a built-in autoresearch loop and write reproducible artifacts."""
    definition = get_loop(loop_name)
    output_path = Path(output_dir).expanduser().resolve()
    if output_dir_has_files(output_path):
        if not overwrite:
            raise ValueError(
                "Autoresearch output directory already exists and is not empty. "
                "Use --overwrite to replace prior artifacts."
            )
        clear_output_dir(output_path)
    else:
        ensure_output_dir(output_path)

    started_at = _utc_now()
    run_id = generate_workflow_run_id()
    budget_timeout = int(
        definition.budget.timeout_seconds
        if timeout_seconds is None
        else timeout_seconds
    )
    selected_models = _normalize_models(loop_name, models)
    trial_limit = int(
        definition.budget.max_trials if max_trials is None else max_trials
    )
    if trial_limit <= 0:
        raise ValueError("--max-trials must be a positive integer.")

    common = {
        "definition": definition,
        "output_path": output_path,
        "run_id": run_id,
        "started_at": started_at,
        "profile": profile,
        "models": selected_models,
        "max_trials": trial_limit,
        "max_trials_explicit": max_trials is not None,
        "timeout_seconds": budget_timeout,
        "dry_run": dry_run,
        "skip_plots": skip_plots,
        "seed": int(seed),
        "dataset": dataset,
    }

    if loop_name == "forecast-daytona":
        return _run_forecast_daytona(**common)
    if loop_name == "classify-daytona":
        return _run_classify_daytona(**common)
    if loop_name == "foundation-gpu-plan":
        return _run_foundation_gpu_plan(**common)
    raise ValueError(
        f"Autoresearch loop '{loop_name}' is registered but has no runner."
    )


def _run_forecast_daytona(**kwargs: Any) -> dict[str, Any]:
    output_path: Path = kwargs["output_path"]
    run_id: str = kwargs["run_id"]
    definition = kwargs["definition"]
    started_at: str = kwargs["started_at"]
    profile: str = kwargs["profile"]
    models: list[str] = kwargs["models"]
    max_trials: int = kwargs["max_trials"]
    max_trials_explicit: bool = kwargs["max_trials_explicit"]
    timeout_seconds: int = kwargs["timeout_seconds"]
    dry_run: bool = kwargs["dry_run"]
    skip_plots: bool = kwargs["skip_plots"]

    dataset_path = _resolve_runtime_file("data/m4_monthly_mini.csv")
    panel = _load_m4_panel(dataset_path)
    series_ids = _forecast_series_for_profile(profile)
    horizon = 18
    season_length = 12
    rolling_origins = _forecast_rolling_origins_for_profile(profile)

    warnings: list[str] = []
    if find_spec("statsforecast") is None:
        unavailable = [method for method in models if method in _STATSFORECAST_METHODS]
        if unavailable:
            warnings.append(
                "statsforecast is not installed; skipping methods that require "
                f"`ts-agents[forecasting]`: {', '.join(unavailable)}."
            )
            models = [
                method for method in models if method not in _STATSFORECAST_METHODS
            ]
    if not models:
        raise ImportError(
            'No forecasting methods are available. Install "ts-agents[forecasting]".'
        )

    trial_specs = _forecast_trial_specs(
        panel=panel,
        series_ids=series_ids,
        horizon=horizon,
        rolling_origins=rolling_origins,
    )
    trial_pairs = _forecast_trial_pairs(trial_specs, models)
    if profile == "full" and not max_trials_explicit:
        max_trials = len(trial_pairs)
    scheduled_pairs = trial_pairs[:max_trials]
    trials: list[dict[str, Any]] = []
    start = time.monotonic()

    for trial_index, (spec, model) in enumerate(scheduled_pairs, start=1):
        if _budget_exhausted(start, timeout_seconds):
            warnings.append(
                f"Stopped before trial {trial_index}; timeout budget exhausted."
            )
            break
        if dry_run:
            trials.append(
                _planned_trial_row(run_id, trial_index, "forecasting", spec, model)
            )
            continue
        trial_timeout = _trial_timeout_for_next(
            start,
            timeout_seconds,
            max_trials=max_trials,
            completed_trials=len(trials),
        )
        with _trial_timeout(trial_timeout):
            trials.append(
                _evaluate_forecast_trial(
                    run_id=run_id,
                    trial_index=trial_index,
                    spec=spec,
                    model=model,
                    horizon=horizon,
                    season_length=season_length,
                )
            )

    ranking = _rank_forecast_trials(trials)
    warnings.extend(_forecast_ranking_warnings(trials))
    best_config = ranking[0] if ranking else {}
    status = (
        "degraded"
        if warnings or any(row.get("status") == "failed" for row in trials)
        else "ok"
    )
    plot_artifacts = _write_optional_plot(
        _write_forecast_plot,
        output_path,
        trials,
        warnings,
        enabled=not skip_plots and not dry_run,
    )
    artifacts = _write_autoresearch_artifacts(
        output_path=output_path,
        loop_name="forecast-daytona",
        run_id=run_id,
        started_at=started_at,
        definition=loop_to_dict(definition),
        options={
            "profile": profile,
            "models": models,
            "max_trials": max_trials,
            "timeout_seconds": timeout_seconds,
            "dry_run": dry_run,
            "horizon": horizon,
            "season_length": season_length,
            "rolling_origins": rolling_origins,
            "dataset": str(dataset_path),
        },
        trials=trials,
        best_config=best_config,
        report=_forecast_report(
            dataset_path=dataset_path,
            models=models,
            trials=trials,
            ranking=ranking,
            dry_run=dry_run,
        ),
        extra_json={"model_ranking": ranking},
        status=status,
        warnings=warnings,
        extra_artifacts=plot_artifacts,
    )

    return _result_payload(
        loop_name="forecast-daytona",
        run_id=run_id,
        status=status,
        summary=f"Forecast autoresearch loop completed with {len(trials)} model-trial rows.",
        output_path=output_path,
        trials=trials,
        best_config=best_config,
        artifacts=artifacts,
        warnings=warnings,
        started_at=started_at,
        data_extra={"ranking": ranking, "dataset": str(dataset_path)},
    )


def _run_classify_daytona(**kwargs: Any) -> dict[str, Any]:
    output_path: Path = kwargs["output_path"]
    run_id: str = kwargs["run_id"]
    definition = kwargs["definition"]
    started_at: str = kwargs["started_at"]
    profile: str = kwargs["profile"]
    models: list[str] = kwargs["models"]
    max_trials: int = kwargs["max_trials"]
    timeout_seconds: int = kwargs["timeout_seconds"]
    dry_run: bool = kwargs["dry_run"]
    skip_plots: bool = kwargs["skip_plots"]
    seed: int = kwargs["seed"]
    dataset: str = kwargs["dataset"]

    dataset_path, dataset_label, dataset_artifacts = _classification_dataset(
        output_path=output_path,
        dataset=dataset,
        seed=seed,
    )
    stream_input = load_labeled_stream_input(
        input_path=str(dataset_path),
        time_col="timestamp",
        value_cols=["x", "y", "z"],
        label_col="label",
    )
    window_sizes = _classification_window_sizes_for_profile(profile)
    n_splits = 5 if profile == "full" else 3
    trials: list[dict[str, Any]] = []
    warnings: list[str] = []
    start = time.monotonic()

    scheduled_models = models[:max_trials]
    for trial_index, classifier in enumerate(scheduled_models, start=1):
        spec = {
            "classifier": classifier,
            "dataset": dataset_label,
            "window_sizes": window_sizes,
        }
        if _budget_exhausted(start, timeout_seconds):
            warnings.append(
                f"Stopped before classifier {classifier}; timeout budget exhausted."
            )
            break
        if dry_run:
            trials.append(
                _planned_trial_row(
                    run_id, trial_index, "classification", spec, classifier
                )
            )
            continue
        trial_timeout = _trial_timeout_for_next(
            start,
            timeout_seconds,
            max_trials=max_trials,
            completed_trials=len(trials),
        )
        with _trial_timeout(trial_timeout):
            trials.append(
                _evaluate_classifier_trial(
                    run_id=run_id,
                    trial_index=trial_index,
                    classifier=classifier,
                    stream_values=stream_input.values,
                    stream_labels=stream_input.labels,
                    window_sizes=window_sizes,
                    n_splits=n_splits,
                    seed=seed,
                )
            )

    ranking = _rank_classification_trials(trials)
    best_config = ranking[0] if ranking else {}
    status = (
        "degraded"
        if warnings or any(row.get("status") == "failed" for row in trials)
        else "ok"
    )
    plot_artifacts = _write_optional_plot(
        _write_classification_plot,
        output_path,
        trials,
        warnings,
        enabled=not skip_plots and not dry_run,
    )
    artifacts = list(dataset_artifacts)
    artifacts.extend(
        _write_autoresearch_artifacts(
            output_path=output_path,
            loop_name="classify-daytona",
            run_id=run_id,
            started_at=started_at,
            definition=loop_to_dict(definition),
            options={
                "profile": profile,
                "models": models,
                "max_trials": max_trials,
                "timeout_seconds": timeout_seconds,
                "dry_run": dry_run,
                "dataset": dataset_label,
                "window_sizes": window_sizes,
                "metric": "balanced_accuracy",
                "balance": "segment_cap",
                "max_windows_per_segment": 25,
                "n_splits": n_splits,
                "seed": seed,
            },
            trials=trials,
            best_config=best_config,
            report=_classification_report(
                dataset_path=dataset_path,
                models=models,
                trials=trials,
                ranking=ranking,
                dry_run=dry_run,
            ),
            extra_json={"model_ranking": ranking},
            status=status,
            warnings=warnings,
            extra_artifacts=plot_artifacts,
        )
    )

    return _result_payload(
        loop_name="classify-daytona",
        run_id=run_id,
        status=status,
        summary=f"Classification autoresearch loop completed with {len(trials)} trials.",
        output_path=output_path,
        trials=trials,
        best_config=best_config,
        artifacts=artifacts,
        warnings=warnings,
        started_at=started_at,
        data_extra={"ranking": ranking, "dataset": str(dataset_path)},
    )


def _run_foundation_gpu_plan(**kwargs: Any) -> dict[str, Any]:
    output_path: Path = kwargs["output_path"]
    run_id: str = kwargs["run_id"]
    definition = kwargs["definition"]
    started_at: str = kwargs["started_at"]
    profile: str = kwargs["profile"]
    models: list[str] = kwargs["models"]
    max_trials: int = kwargs["max_trials"]
    timeout_seconds: int = kwargs["timeout_seconds"]

    plan = _foundation_gpu_plan(
        run_id=run_id,
        profile=profile,
        models=models,
        max_trials=max_trials,
        timeout_seconds=timeout_seconds,
    )
    chronos_config = _chronos_config_yaml(plan)
    moment_config = _moment_config_yaml(plan)
    commands = _foundation_commands(plan)
    report = _foundation_report(plan)

    artifacts = [
        _write_json(output_path / "foundation_gpu_plan.json", plan, "Foundation-model plan."),
        _write_text(
            output_path / "chronos_finetune_config.yaml",
            chronos_config,
            "Chronos fine-tuning config template.",
            mime_type="text/yaml",
        ),
        _write_text(
            output_path / "moment_classification_config.yaml",
            moment_config,
            "MOMENT classification fine-tuning config template.",
            mime_type="text/yaml",
        ),
        _write_text(
            output_path / "commands.sh",
            commands,
            "Plan-only GPU setup notes and validation commands.",
            mime_type="text/x-shellscript",
        ),
        _write_text(output_path / "report.md", report, "Foundation GPU plan report."),
    ]
    commands_path = output_path / "commands.sh"
    commands_path.chmod(commands_path.stat().st_mode | stat.S_IXUSR)

    best_config: dict[str, Any] = {}
    manifest = _manifest_payload(
        loop_name="foundation-gpu-plan",
        run_id=run_id,
        status="plan-only",
        summary="Foundation GPU plan materialized; no model training or evaluation was run.",
        output_path=output_path,
        started_at=started_at,
        definition=loop_to_dict(definition),
        options={
            "profile": profile,
            "models": models,
            "max_trials": max_trials,
            "timeout_seconds": timeout_seconds,
            "gpu": "RTX PRO 6000 Blackwell",
            "plan_only": True,
        },
        artifacts=artifacts,
        warnings=[],
        best_config=best_config,
    )
    artifacts.append(_write_json(output_path / WORKFLOW_MANIFEST_FILENAME, manifest, "Autoresearch run manifest."))

    return _result_payload(
        loop_name="foundation-gpu-plan",
        run_id=run_id,
        status="plan-only",
        summary="Foundation GPU plan materialized; no model training or evaluation was run.",
        output_path=output_path,
        trials=[],
        best_config=best_config,
        artifacts=artifacts,
        warnings=[],
        started_at=started_at,
        data_extra={
            "plan": plan,
            "plan_only": True,
            "recommended_runs": plan["recommended_runs"],
            "quality_flags": ["plan_only"],
        },
    )

def _normalize_models(loop_name: str, models: Optional[Iterable[str]]) -> list[str]:
    if models is None:
        if loop_name == "forecast-daytona":
            return list(DEFAULT_FORECAST_METHODS)
        if loop_name == "classify-daytona":
            return list(DEFAULT_CLASSIFIERS)
        if loop_name == "foundation-gpu-plan":
            return list(DEFAULT_FOUNDATION_MODELS)
        return []
    normalized = []
    for model in models:
        if model is None:
            raise ValueError(f"Model list for {loop_name} contains a null entry.")
        value = str(model).strip()
        if value:
            normalized.append(value)
    if not normalized:
        return _normalize_models(loop_name, None)
    valid = {
        "forecast-daytona": set(DEFAULT_FORECAST_METHODS),
        "classify-daytona": set(DEFAULT_CLASSIFIERS),
        "foundation-gpu-plan": set(DEFAULT_FOUNDATION_MODELS),
    }.get(loop_name)
    if valid is not None:
        invalid = sorted(set(normalized).difference(valid))
        if invalid:
            raise ValueError(
                f"Unsupported model(s) for {loop_name}: {', '.join(invalid)}."
            )
    return normalized


def _resolve_runtime_file(relative_path: str) -> Path:
    resolved = resolve_existing_path(relative_path)
    path = resolved if resolved is not None else Path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Required autoresearch dataset not found: {relative_path}")
    return path.resolve()


def _load_m4_panel(dataset_path: Path) -> dict[str, dict[str, np.ndarray]]:
    df = pd.read_csv(dataset_path)
    required = {"unique_id", "split", "ds", "y"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{dataset_path} is missing required columns: {missing}")
    panel: dict[str, dict[str, np.ndarray]] = {}
    for series_id, series_df in df.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        train = series_df[series_df["split"] == "train"]["y"].to_numpy(dtype=float)
        holdout = series_df[series_df["split"] == "holdout"]["y"].to_numpy(dtype=float)
        panel[str(series_id)] = {"train": train, "holdout": holdout}
    return panel


def _forecast_series_for_profile(profile: str) -> list[str]:
    if profile == "smoke":
        return DEFAULT_FORECAST_SERIES[:1]
    return list(DEFAULT_FORECAST_SERIES)


def _forecast_rolling_origins_for_profile(profile: str) -> int:
    if profile == "smoke":
        return 1
    if profile == "full":
        return 3
    return 2


def _classification_window_sizes_for_profile(profile: str) -> list[int]:
    if profile == "smoke":
        return [32, 64]
    if profile == "full":
        return [32, 64, 96, 128, 160, 192, 224]
    return list(DEFAULT_WINDOW_SIZES)


def _forecast_trial_specs(
    *,
    panel: dict[str, dict[str, np.ndarray]],
    series_ids: list[str],
    horizon: int,
    rolling_origins: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for series_id in series_ids:
        if series_id not in panel:
            continue
        train = panel[series_id]["train"]
        holdout = panel[series_id]["holdout"]
        for origin_index, origin in enumerate(_rolling_origins(len(train), horizon, rolling_origins), start=1):
            specs.append(
                {
                    "series_id": series_id,
                    "phase": f"rolling_origin_{origin_index}",
                    "origin": int(origin),
                    "train": train[:origin],
                    "actual": train[origin:origin + horizon],
                }
            )
        specs.append(
            {
                "series_id": series_id,
                "phase": "holdout",
                "origin": int(len(train)),
                "train": train,
                "actual": holdout[:horizon],
            }
        )
    return specs


def _rolling_origins(train_length: int, horizon: int, n_origins: int) -> list[int]:
    if n_origins <= 0 or train_length < horizon * 2:
        return []
    latest_origin = train_length - horizon
    first_origin = latest_origin - horizon * (n_origins - 1)
    origins = [first_origin + i * horizon for i in range(n_origins)]
    return [
        int(origin)
        for origin in origins
        if origin >= horizon and origin + horizon <= train_length
    ]


def _forecast_trial_pairs(
    trial_specs: list[dict[str, Any]],
    models: list[str],
) -> list[tuple[dict[str, Any], str]]:
    return [(spec, model) for spec in trial_specs for model in models]


def _evaluate_forecast_trial(
    *,
    run_id: str,
    trial_index: int,
    spec: dict[str, Any],
    model: str,
    horizon: int,
    season_length: int,
) -> dict[str, Any]:
    actual = np.asarray(spec["actual"], dtype=float)
    started = time.monotonic()
    row = {
        "run_id": run_id,
        "trial_id": f"forecast-{trial_index:03d}-{model}",
        "trial_index": trial_index,
        "task": "forecasting",
        "dataset": "m4_monthly_mini",
        "series_id": spec["series_id"],
        "phase": spec["phase"],
        "origin": spec["origin"],
        "model": model,
    }
    try:
        forecast = _forecast_with_method(
            model,
            np.asarray(spec["train"], dtype=float),
            horizon=horizon,
            season_length=season_length,
        )[: len(actual)]
        metrics = _forecast_metrics(actual, forecast)
        row.update(metrics)
        row["status"] = "ok"
    except Exception as exc:
        row.update({
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
    row["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return row


def _forecast_with_method(method: str, series: np.ndarray, *, horizon: int, season_length: int) -> np.ndarray:
    from ts_agents.core.forecasting import (
        forecast_arima,
        forecast_ets,
        forecast_seasonal_naive,
        forecast_theta,
    )

    func_map = {
        "seasonal_naive": forecast_seasonal_naive,
        "theta": forecast_theta,
        "ets": forecast_ets,
        "arima": forecast_arima,
    }
    return np.asarray(
        func_map[method](series, horizon=horizon, season_length=season_length).forecast,
        dtype=float,
    )


def _forecast_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    errors = actual - forecast
    denominator = np.abs(actual) + np.abs(forecast)
    smape_values = np.divide(
        200.0 * np.abs(errors),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100.0)
        if np.any(nonzero)
        else float("nan")
    )
    return {
        "smape": float(np.mean(smape_values)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": mape,
    }


def _classification_dataset(
    *,
    output_path: Path,
    dataset: str,
    seed: int,
) -> tuple[Path, str, list[ArtifactRef]]:
    if dataset == "synthetic":
        return _write_synthetic_classification_dataset(output_path, seed)
    wisdm = resolve_existing_path("data/wisdm_subset.csv")
    if dataset in {"auto", "wisdm_subset"} and wisdm is not None and wisdm.exists():
        return wisdm.resolve(), "wisdm_subset", []
    return _write_synthetic_classification_dataset(output_path, seed)


def _write_synthetic_classification_dataset(output_path: Path, seed: int) -> tuple[Path, str, list[ArtifactRef]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    timestamp = 0
    specs = [
        ("idle", 5, 0.05, 0.0),
        ("walk", 5, 0.15, 1.6),
        ("jog", 5, 0.22, 2.8),
    ]
    for repeat in range(8):
        for label, seconds, noise, freq in specs:
            n = seconds * 20
            t = np.arange(n) / 20.0
            if label == "idle":
                values = rng.normal(0.0, noise, size=(n, 3))
            else:
                phase = rng.uniform(0.0, 2 * np.pi, size=3)
                amp = 1.0 if label == "walk" else 1.8
                values = np.column_stack(
                    [
                        amp * np.sin(2 * np.pi * freq * t + phase[0]),
                        0.8 * amp * np.sin(2 * np.pi * freq * t + phase[1]),
                        0.6 * amp * np.sin(2 * np.pi * freq * t + phase[2]),
                    ]
                )
                values += rng.normal(0.0, noise, size=values.shape)
            for x, y, z in values:
                rows.append({"timestamp": timestamp, "x": x, "y": y, "z": z, "label": label})
                timestamp += 1
    path = output_path / "synthetic_labeled_stream.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return (
        path.resolve(),
        "synthetic_gait",
        [artifact_ref(
            kind="csv",
            path=path,
            mime_type="text/csv",
            description="Generated synthetic labeled stream used for classification.",
            created_by="classify-daytona",
        )],
    )


def _evaluate_classifier_trial(
    *,
    run_id: str,
    trial_index: int,
    classifier: str,
    stream_values: np.ndarray,
    stream_labels: np.ndarray,
    window_sizes: list[int],
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {
        "run_id": run_id,
        "trial_id": f"classify-{trial_index:03d}-{classifier}",
        "trial_index": trial_index,
        "task": "classification",
        "dataset": "labeled_activity_stream",
        "model": classifier,
        "classifier": classifier,
    }
    try:
        from ts_agents.core.windowing import select_window_size

        selection = select_window_size(
            stream_values,
            stream_labels,
            window_sizes=window_sizes,
            metric="balanced_accuracy",
            classifier=classifier,
            labeling="strict",
            balance="segment_cap",
            max_windows_per_segment=25,
            n_splits=n_splits,
            test_size=0.25,
            seed=seed,
        )
        best_window_size = int(selection.best_window_size)
        score = _as_optional_float(selection.scores_by_window.get(best_window_size))
        row.update(
            {
                "status": "ok",
                "best_window_size": best_window_size,
                "balanced_accuracy": score,
                "accuracy": None,
                "f1_macro": None,
                "n_windows": int(
                    selection.n_windows_by_window.get(best_window_size, 0)
                ),
                "effective_backend": classifier,
                "evaluation_stage": "window_selection_cv",
                "selection_scores_by_window": {
                    str(window): _as_optional_float(value)
                    for window, value in selection.scores_by_window.items()
                },
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        )
    row["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return row


def _as_optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return None


def _rank_forecast_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [row for row in trials if row.get("status") == "ok"]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in ok:
        by_model.setdefault(str(row["model"]), []).append(row)
    ranking = []
    for model, rows in by_model.items():
        holdout_rows = [row for row in rows if row.get("phase") == "holdout"]
        rolling_rows = [row for row in rows if str(row.get("phase", "")).startswith("rolling_origin_")]
        ranking.append(
            {
                "model": model,
                "primary_metric": "smape",
                "smape": _mean_metric(rows, "smape"),
                "holdout_smape": _mean_metric(holdout_rows, "smape"),
                "rolling_smape": _mean_metric(rolling_rows, "smape"),
                "mae": _mean_metric(rows, "mae"),
                "rmse": _mean_metric(rows, "rmse"),
                "n_trials": len(rows),
                "n_holdout_trials": len(holdout_rows),
                "n_rolling_trials": len(rolling_rows),
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            _sort_low_metric(row.get("smape")),
            _sort_low_metric(row.get("mae")),
            _sort_low_metric(row.get("rmse")),
            str(row.get("model", "")),
        ),
    )


def _forecast_ranking_warnings(trials: list[dict[str, Any]]) -> list[str]:
    ok = [row for row in trials if row.get("status") == "ok"]
    if not ok:
        return []
    warnings: list[str] = []
    if not any(row.get("phase") == "holdout" for row in ok):
        warnings.append("Forecast ranking has no successful holdout rows; ranking uses rolling-origin rows only.")
    counts: dict[str, int] = {}
    for row in ok:
        counts[str(row.get("model"))] = counts.get(str(row.get("model")), 0) + 1
    if len(set(counts.values())) > 1:
        warnings.append(
            "Forecast ranking is based on an unequal number of successful rows per model: "
            + ", ".join(f"{model}={count}" for model, count in sorted(counts.items()))
            + "."
        )
    return warnings


def _rank_classification_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [row for row in trials if row.get("status") == "ok"]
    ranking = [
        {
            "model": row["model"],
            "primary_metric": "balanced_accuracy",
            "balanced_accuracy": row.get("balanced_accuracy"),
            "best_window_size": row.get("best_window_size"),
            "n_windows": row.get("n_windows"),
            "n_trials": 1,
        }
        for row in ok
    ]
    return sorted(
        ranking,
        key=lambda row: (
            _sort_high_metric(row.get("balanced_accuracy")),
            _sort_low_metric(row.get("best_window_size")),
            _sort_high_metric(row.get("n_windows")),
            str(row.get("model", "")),
        ),
    )


def _sort_low_metric(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isfinite(number):
            return number
    return float("inf")


def _sort_high_metric(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isfinite(number):
            return -number
    return float("inf")


def _mean_metric(rows: list[dict[str, Any]], key: str) -> Optional[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        number = float(value)
        if np.isfinite(number):
            values.append(number)
    return float(np.mean(values)) if values else None


def _write_autoresearch_artifacts(
    *,
    output_path: Path,
    loop_name: str,
    run_id: str,
    started_at: str,
    definition: dict[str, Any],
    options: dict[str, Any],
    trials: list[dict[str, Any]],
    best_config: dict[str, Any],
    report: str,
    extra_json: Optional[dict[str, Any]] = None,
    status: str = "ok",
    warnings: Optional[list[str]] = None,
    extra_artifacts: Optional[list[ArtifactRef]] = None,
) -> list[ArtifactRef]:
    trials_df = pd.DataFrame(trials)
    artifacts = [
        _write_csv(output_path / "trials.csv", trials_df, "Autoresearch trial table."),
        _write_jsonl(output_path / "trials.jsonl", trials, "Autoresearch trial records."),
        _write_json(output_path / "best_config.json", best_config, "Best autoresearch configuration."),
        _write_text(output_path / "report.md", report, "Autoresearch markdown report."),
    ]
    if extra_json:
        artifacts.append(_write_json(output_path / "summary.json", extra_json, "Autoresearch loop summary."))
    artifacts.extend(extra_artifacts or [])
    manifest = _manifest_payload(
        loop_name=loop_name,
        run_id=run_id,
        status=status,
        summary=f"{loop_name} autoresearch artifacts.",
        output_path=output_path,
        started_at=started_at,
        definition=definition,
        options=options,
        artifacts=artifacts,
        warnings=warnings or [],
        best_config=best_config,
    )
    artifacts.append(_write_json(output_path / WORKFLOW_MANIFEST_FILENAME, manifest, "Autoresearch run manifest."))
    return artifacts


def _write_json(path: Path, data: Any, description: str) -> ArtifactRef:
    payload = render_output(to_jsonable(data), json_output=True)
    write_output(payload, str(path))
    return artifact_ref(
        kind="json",
        path=path,
        mime_type="application/json",
        description=description,
        created_by="autoresearch",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]], description: str) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(to_jsonable(row), sort_keys=True) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")
    return artifact_ref(
        kind="jsonl",
        path=path,
        mime_type="application/x-jsonlines",
        description=description,
        created_by="autoresearch",
    )


def _write_csv(path: Path, dataframe: pd.DataFrame, description: str) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return artifact_ref(
        kind="csv",
        path=path,
        mime_type="text/csv",
        description=description,
        created_by="autoresearch",
    )


def _write_text(path: Path, content: str, description: str, *, mime_type: str = "text/markdown") -> ArtifactRef:
    write_output(content, str(path))
    return artifact_ref(
        kind="markdown" if mime_type == "text/markdown" else "text",
        path=path,
        mime_type=mime_type,
        description=description,
        created_by="autoresearch",
    )


def _manifest_payload(
    *,
    loop_name: str,
    run_id: str,
    status: str,
    summary: str,
    output_path: Path,
    started_at: str,
    definition: dict[str, Any],
    options: dict[str, Any],
    artifacts: list[ArtifactRef],
    warnings: list[str],
    best_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "kind": "autoresearch_run",
        "loop": loop_name,
        "run_id": run_id,
        "status": status,
        "summary": summary,
        "created_at": started_at,
        "output_dir": str(output_path),
        "manifest_path": str(output_path / WORKFLOW_MANIFEST_FILENAME),
        "definition": definition,
        "options": to_jsonable(options),
        "best_config": to_jsonable(best_config),
        "warnings": to_jsonable(warnings),
        "artifacts": [to_jsonable(artifact) for artifact in artifacts],
    }


def _result_payload(
    *,
    loop_name: str,
    run_id: str,
    status: str,
    summary: str,
    output_path: Path,
    trials: list[dict[str, Any]],
    best_config: dict[str, Any],
    artifacts: list[ArtifactRef],
    warnings: list[str],
    started_at: str,
    data_extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    data = {
        "loop": loop_name,
        "run_id": run_id,
        "output_dir": str(output_path),
        "manifest_path": str(output_path / WORKFLOW_MANIFEST_FILENAME),
        "started_at": started_at,
        "trial_count": len(trials),
        "best_config": best_config,
        "quality_flags": [],
    }
    if data_extra:
        data.update(data_extra)
    return {
        "kind": "autoresearch",
        "summary": summary,
        "status": status,
        "data": data,
        "artifacts": artifacts,
        "warnings": warnings,
    }


def _planned_trial_row(
    run_id: str,
    trial_index: int,
    task: str,
    spec: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "trial_id": f"{task}-{trial_index:03d}-{model}",
        "trial_index": trial_index,
        "task": task,
        "model": model,
        "status": "planned",
        "spec": to_jsonable({k: v for k, v in spec.items() if k not in {"train", "actual"}}),
    }


def _budget_exhausted(start: float, timeout_seconds: int) -> bool:
    return timeout_seconds > 0 and (time.monotonic() - start) >= timeout_seconds


def _trial_timeout_for_next(
    start: float,
    timeout_seconds: int,
    *,
    max_trials: int,
    completed_trials: int,
) -> Optional[float]:
    if timeout_seconds <= 0:
        return None
    remaining = timeout_seconds - (time.monotonic() - start)
    if remaining <= 0:
        return 0.001
    nominal = timeout_seconds / max(1, max_trials)
    per_trial_cap = max(30.0, nominal * 2.0)
    return max(0.001, min(remaining, per_trial_cap))


@contextmanager
def _trial_timeout(seconds: Optional[float]):
    if (
        seconds is None
        or seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "ITIMER_REAL")
    ):
        yield
        return

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"Trial exceeded {seconds:.1f}s wall-clock limit.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    armed_at = time.monotonic()
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        elapsed = time.monotonic() - armed_at
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            remaining = previous_timer[0] - elapsed
            if remaining <= 0:
                signal.raise_signal(signal.SIGALRM)
            else:
                signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_optional_plot(
    plot_writer: Any,
    output_path: Path,
    trials: list[dict[str, Any]],
    warnings: list[str],
    *,
    enabled: bool,
) -> list[ArtifactRef]:
    if not enabled:
        return []
    try:
        return plot_writer(output_path, trials)
    except Exception as exc:
        warnings.append(f"Skipped ranking plot because {type(exc).__name__}: {exc}")
        return []


def _forecast_report(
    *,
    dataset_path: Path,
    models: list[str],
    trials: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    dry_run: bool,
) -> str:
    lines = [
        "# Forecast Daytona Autoresearch",
        "",
        f"- dataset: `{dataset_path}`",
        f"- models: `{', '.join(models)}`",
        f"- mode: `{'dry-run' if dry_run else 'executed'}`",
        "- primary metric: `sMAPE` (lower is better)",
        "",
        "## Ranking",
        "",
    ]
    if ranking:
        lines.extend(_markdown_table(ranking, ["model", "smape", "mae", "rmse", "n_trials"]))
    else:
        lines.append("No completed model trials were available for ranking.")
    lines.extend(["", "## Trial Count", "", f"- rows: `{len(trials)}`", ""])
    return "\n".join(lines)


def _classification_report(
    *,
    dataset_path: Path,
    models: list[str],
    trials: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    dry_run: bool,
) -> str:
    lines = [
        "# Classification Daytona Autoresearch",
        "",
        f"- dataset: `{dataset_path}`",
        f"- classifiers: `{', '.join(models)}`",
        f"- mode: `{'dry-run' if dry_run else 'executed'}`",
        "- primary metric: `balanced_accuracy` (higher is better)",
        "",
        "## Ranking",
        "",
    ]
    if ranking:
        lines.extend(_markdown_table(ranking, ["model", "balanced_accuracy", "best_window_size", "n_windows"]))
    else:
        lines.append("No completed classifier trials were available for ranking.")
    lines.extend(["", "## Trial Count", "", f"- rows: `{len(trials)}`", ""])
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_forecast_plot(output_path: Path, trials: list[dict[str, Any]]) -> list[ArtifactRef]:
    ok = [row for row in trials if row.get("status") == "ok" and isinstance(row.get("smape"), (int, float))]
    if not ok:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    df = pd.DataFrame(ok)
    summary = df.groupby("model", as_index=False)["smape"].mean().sort_values("smape")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["model"], summary["smape"])
    ax.set_ylabel("mean sMAPE")
    ax.set_title("Forecast autoresearch ranking")
    fig.tight_layout()
    path = output_path / "ranking.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [artifact_ref(kind="image", path=path, mime_type="image/png", description="Forecast ranking plot.", created_by="autoresearch")]


def _write_classification_plot(output_path: Path, trials: list[dict[str, Any]]) -> list[ArtifactRef]:
    ok = [row for row in trials if row.get("status") == "ok" and isinstance(row.get("balanced_accuracy"), (int, float))]
    if not ok:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    df = pd.DataFrame(ok).sort_values("balanced_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["model"], df["balanced_accuracy"])
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Classification autoresearch ranking")
    fig.tight_layout()
    path = output_path / "ranking.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [artifact_ref(kind="image", path=path, mime_type="image/png", description="Classification ranking plot.", created_by="autoresearch")]


def _foundation_gpu_plan(
    *,
    run_id: str,
    profile: str,
    models: list[str],
    max_trials: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    model_revisions = {
        "amazon/chronos-2": "254b5357164a84326913b0695216f690752ac55d",
        "amazon/chronos-t5-small": "4753ebecb99f65f84cd2823c56f7ab22b02ac303",
        "AutonLab/MOMENT-1-large": "3582f9d7f033eea9d43e6a802ba0e36d5f26b57c",
    }
    recommended_runs = []
    if "amazon/chronos-2" in models:
        recommended_runs.append(
            {
                "task": "forecasting",
                "model": "amazon/chronos-2",
                "revision": model_revisions["amazon/chronos-2"],
                "mode": "zero_shot_evaluation",
                "primary_metric": "sMAPE",
            }
        )
    if "amazon/chronos-t5-small" in models:
        recommended_runs.append(
            {
                "task": "forecasting",
                "model": "amazon/chronos-t5-small",
                "revision": model_revisions["amazon/chronos-t5-small"],
                "mode": "fine_tune_fallback_after_adapter",
                "primary_metric": "sMAPE",
            }
        )
    if "AutonLab/MOMENT-1-large" in models:
        recommended_runs.append(
            {
                "task": "classification",
                "model": "AutonLab/MOMENT-1-large",
                "revision": model_revisions["AutonLab/MOMENT-1-large"],
                "mode": "linear_probe_then_peft_after_adapter",
                "primary_metric": "balanced_accuracy",
            }
        )

    return {
        "run_id": run_id,
        "profile": profile,
        "status": "plan-only",
        "plan_only": True,
        "target_hardware": "1x RTX PRO 6000 Blackwell",
        "selected_models": models,
        "model_revisions": model_revisions,
        "budget": {
            "timeout_seconds": timeout_seconds,
            "max_trials": max_trials,
            "smoke": {"minutes": 30, "seeds": [1337], "max_steps": 1000},
            "full": {"hours": 4, "seeds": [1337, 2027, 9001], "checkpoint_cap_gb": 5},
        },
        "forecasting": {
            "primary_model": "amazon/chronos-2",
            "primary_model_revision": model_revisions["amazon/chronos-2"],
            "finetune_model": "amazon/chronos-t5-small",
            "finetune_model_revision": model_revisions["amazon/chronos-t5-small"],
            "package": "chronos-forecasting>=2.0,<3",
            "dataset": "data/m4_monthly_mini.csv; adapter must create GluonTS Arrow before fine-tuning",
            "metrics": ["sMAPE", "MASE", "MAE", "RMSE"],
            "source": "https://github.com/amazon-science/chronos-forecasting",
        },
        "classification": {
            "primary_model": "AutonLab/MOMENT-1-large",
            "primary_model_revision": model_revisions["AutonLab/MOMENT-1-large"],
            "package": "momentfm",
            "dataset": "generated/vendored labeled stream; adapter must create windowed_activity_dataset.npz",
            "metrics": ["balanced_accuracy", "macro_f1", "accuracy"],
            "source": "https://github.com/moment-timeseries-foundation-model/moment",
        },
        "adapter_requirements": [
            "Create m4_monthly_mini.arrow before Chronos fine-tuning.",
            "Create windowed_activity_dataset.npz before MOMENT classification fine-tuning.",
            "Pin package versions in the training environment lockfile before running fine-tuning.",
        ],
        "recommended_runs": recommended_runs,
    }


def _chronos_config_yaml(plan: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Chronos fine-tuning template for ts-agents foundation-gpu-plan",
            "# Plan-only artifact: adapter outputs must be materialized before use.",
            "model_id: amazon/chronos-t5-small",
            f"model_revision: {plan['model_revisions']['amazon/chronos-t5-small']}",
            "random_init: false",
            "tokenizer_type: mean_scale_uniform_bins",
            "context_length: 512",
            "prediction_length: 18",
            "max_steps: 1000",
            "learning_rate: 0.001",
            "per_device_train_batch_size: 32",
            "torch_compile: false",
            "output_dir: outputs/autoresearch/foundation-gpu-plan/chronos",
            "training_data_paths: []",
            "validation_data_paths: []",
            "probability: []",
            "adapter_required: true",
            f"source: {plan['forecasting']['source']}",
            "",
        ]
    )


def _moment_config_yaml(plan: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MOMENT classification fine-tuning template for ts-agents foundation-gpu-plan",
            "# Plan-only artifact: adapter outputs must be materialized before use.",
            "model_id: AutonLab/MOMENT-1-large",
            f"model_revision: {plan['model_revisions']['AutonLab/MOMENT-1-large']}",
            "task_name: classification",
            "strategy: linear_probe_then_peft",
            "precision: bf16",
            "max_epochs: 5",
            "early_stopping_patience: 2",
            "batch_size: 64",
            "learning_rate: 0.0001",
            "checkpoint_cap_gb: 5",
            "dataset_path: null",
            "adapter_required: true",
            f"source: {plan['classification']['source']}",
            "",
        ]
    )


def _foundation_commands(plan: dict[str, Any]) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "echo 'foundation-gpu-plan is plan-only; no training or evaluation is launched by this script.'",
            "echo 'Materialize adapters and lock package versions before using the upstream training recipes.'",
            "",
            "# Suggested GPU setup once the plan is promoted to an executable training run:",
            "# python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision",
            "# python -m pip install 'chronos-forecasting>=2.0,<3' momentfm",
            "",
            "python - <<'PY'",
            "from pathlib import Path",
            "required = [",
            "    Path('outputs/autoresearch/foundation-gpu-plan/m4_monthly_mini.arrow'),",
            "    Path('outputs/autoresearch/foundation-gpu-plan/windowed_activity_dataset.npz'),",
            "]",
            "missing = [str(path) for path in required if not path.exists()]",
            "if missing:",
            "    print('adapter outputs not present yet: ' + ', '.join(missing))",
            "else:",
            "    print('adapter outputs are present')",
            "PY",
            "",
            f"# run_id: {plan['run_id']}",
            "",
        ]
    )


def _foundation_report(plan: dict[str, Any]) -> str:
    revision_lines = [
        f"- `{model}` revision `{revision}`"
        for model, revision in plan["model_revisions"].items()
    ]
    lines = [
        "# Foundation GPU Plan",
        "",
        "This loop is plan-only. It materializes pinned model choices, config templates,",
        "and adapter requirements, but it does not train, fine-tune, or evaluate models.",
        "",
        f"- target hardware: `{plan['target_hardware']}`",
        "- forecasting foundation model: `amazon/chronos-2`",
        "- forecasting fine-tune fallback: `amazon/chronos-t5-small`",
        "- classification foundation model: `AutonLab/MOMENT-1-large`",
        "",
        "## Revisions",
        "",
    ]
    lines.extend(revision_lines)
    lines.extend(
        [
            "",
            "## Budget",
            "",
            f"- timeout seconds: `{plan['budget']['timeout_seconds']}`",
            f"- max planned runs: `{plan['budget']['max_trials']}`",
            "- smoke: 30 minutes, one seed, up to 1000 steps",
            "- full: 4 hours, three seeds, bf16, checkpoint cap 5 GiB",
            "",
            "## Required Adapters",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in plan["adapter_requirements"])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `foundation_gpu_plan.json`",
            "- `chronos_finetune_config.yaml`",
            "- `moment_classification_config.yaml`",
            "- `commands.sh`",
            "",
        ]
    )
    return "\n".join(lines)
