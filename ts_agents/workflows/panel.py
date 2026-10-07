"""Global GBM and neural panel forecasting with explicit rolling validation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from contextlib import redirect_stdout
import sys
import time

import numpy as np
import pandas as pd

from ts_agents.cli.input_parsing import PanelInput
from ts_agents.contracts import ToolPayload
from ts_agents.core.forecasting.panel import (
    METHOD_DEPENDENCIES,
    PanelBackend,
    normalize_panel,
    score_predictions,
)
from .common import (
    artifact_ref,
    attach_workflow_run_metadata,
    ensure_output_dir,
    write_dataframe_artifact,
    write_json_artifact,
    write_plot_artifact,
    write_text_artifact,
)


def run_forecast_panel_workflow(
    panel_input: PanelInput,
    *,
    output_dir: str,
    freq: str,
    horizon: int = 24,
    methods=None,
    season_length: int = 24,
    n_windows: int = 3,
    step_size=None,
    lags=None,
    n_estimators: int = 200,
    max_steps: int = 1000,
    input_size=None,
    num_threads: int = 2,
    accelerator: str = "cpu",
    seed: int = 1337,
    skip_plots=False,
    run_id=None,
    resumed=False,
    output_dir_mode="explicit",
    defer_finalization=False,
) -> ToolPayload:
    """Refit each method at each chronological origin, then fit for future forecasts.

    Scores are validation scores used to rank models, not an independent final
    test. Callers must keep their final test targets outside this input.
    """
    methods = list(methods or ["seasonal_naive", "lightgbm"])
    if (
        not methods
        or len(set(methods)) != len(methods)
        or set(methods) - set(METHOD_DEPENDENCIES)
    ):
        raise ValueError(
            "Choose distinct methods from seasonal_naive, lightgbm, histgbm, nhits."
        )
    step_size = horizon if step_size is None else step_size
    input_size = 2 * horizon if input_size is None else input_size
    lags = list(lags if lags is not None else [1, season_length, 7 * season_length])
    if any(
        value <= 0
        for value in (
            horizon,
            season_length,
            n_windows,
            step_size,
            input_size,
            n_estimators,
            max_steps,
            num_threads,
        )
    ):
        raise ValueError(
            "Horizons, periods, windows, steps, and model budgets must be positive."
        )
    if not lags or any(lag <= 0 for lag in lags) or len(set(lags)) != len(lags):
        raise ValueError("Lags must be distinct positive integers.")
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError("Choose accelerator cpu or gpu explicitly.")
    frame = normalize_panel(pd.DataFrame(panel_input.records), freq)
    held_rows = horizon + (n_windows - 1) * step_size
    minimum = (
        season_length + 1
    )  # train-only MASE needs at least one seasonal difference
    if set(methods) & {"lightgbm", "histgbm"}:
        minimum = max(minimum, max(lags) + 2)
    if "nhits" in methods:
        minimum = max(minimum, input_size + 2 * horizon)
    if frame.groupby("unique_id").size().min() - held_rows < minimum:
        raise ValueError(
            f"Each series needs at least {held_rows + minimum} rows for these models/backtests."
        )
    missing = sorted(
        {
            dep
            for method in methods
            for dep in METHOD_DEPENDENCIES[method]
            if find_spec(dep) is None
        }
    )
    if missing:
        extras = [
            extra
            for extra, names in (("ml", {"lightgbm", "histgbm"}), ("neural", {"nhits"}))
            if set(methods) & names
        ]
        raise ImportError(
            f"Missing panel dependencies: {', '.join(missing)}. Install ts-agents[{','.join(extras)}]."
        )
    output = ensure_output_dir(output_dir)
    config = dict(
        lags=lags,
        n_estimators=n_estimators,
        max_steps=max_steps,
        input_size=input_size,
        num_threads=num_threads,
        accelerator=accelerator,
        seed=seed,
    )
    options = dict(
        freq=freq,
        horizon=horizon,
        methods=methods,
        season_length=season_length,
        n_windows=n_windows,
        step_size=step_size,
        skip_plots=skip_plots,
        **config,
    )
    scored, runtime = [], []
    # Common end dates ensure that every panel member is trained only on the
    # same information cutoff. Longer series can retain their older history.
    for window in range(n_windows):
        tail = horizon + (n_windows - window - 1) * step_size
        train = frame.groupby("unique_id", group_keys=False).head(-tail).copy()
        targets = frame.groupby("unique_id", group_keys=False).tail(tail)
        actual = targets.groupby("unique_id", group_keys=False).head(horizon)[
            ["unique_id", "ds", "y"]
        ]
        cutoff = train.ds.max()
        for method in methods:
            started = time.monotonic()
            backend = PanelBackend(method, freq, season_length, config)
            with redirect_stdout(sys.stderr):
                backend.fit(train, horizon)
                predicted = backend.predict(horizon)
            rows = score_predictions(actual, predicted, method, train, season_length)
            rows["cutoff"] = cutoff
            rows["window"] = window
            rows["horizon_step"] = (
                rows.groupby("unique_id").ds.rank(method="dense").astype(int)
            )
            scored.append(rows)
            runtime.append(
                dict(model=method, window=window, seconds=time.monotonic() - started)
            )
    predictions = pd.concat(scored, ignore_index=True)
    metrics = _metrics(predictions, ["model"])
    per_series = _metrics(predictions, ["model", "unique_id"])
    per_horizon = _metrics(predictions, ["model", "horizon_step"])
    best = min(metrics, key=lambda row: (row["rmse"], row["model"]))["model"]
    artifacts = []
    forecasts = []
    for method in methods:
        started = time.monotonic()
        backend = PanelBackend(method, freq, season_length, config)
        with redirect_stdout(sys.stderr):
            backend.fit(frame, horizon)
            future = backend.predict(horizon).rename(columns={method: "prediction"})
        expected = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": series_id,
                        "ds": pd.date_range(
                            group.ds.iloc[-1], periods=horizon + 1, freq=freq
                        )[1:],
                        "y": 0.0,
                    }
                )
                for series_id, group in frame.groupby("unique_id")
            ],
            ignore_index=True,
        )
        # Reuse coverage validation, without treating artificial zeros as scores.
        score_predictions(
            expected,
            future.rename(columns={"prediction": method}),
            method,
            frame,
            season_length,
        )
        future["model"] = method
        forecasts.append(future)
        directory = output / "models" / method
        with redirect_stdout(sys.stderr):
            backend.save(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                artifacts.append(
                    artifact_ref(
                        kind="model",
                        path=path,
                        created_by="forecast-panel",
                        mime_type="application/octet-stream",
                        description=f"Native {method} saved model/history; load only trusted artifacts.",
                    )
                )
        runtime.append(
            dict(model=method, window="final_fit", seconds=time.monotonic() - started)
        )
    flags = []
    warnings = []
    if (predictions.mase_scale == 0).any():
        flags.append("mase_undefined_for_constant_training_series")
        warnings.append(
            "Some training seasonal scales are zero; their MASE is undefined and omitted from aggregates."
        )
    if step_size < horizon:
        warnings.append(
            "Validation horizons overlap; repeated target timestamps represent different forecast origins."
        )
    packages = {}
    for package in (
        "ts-agents",
        "mlforecast",
        "lightgbm",
        "scikit-learn",
        "neuralforecast",
        "torch",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    summary = dict(
        options=options,
        metrics=metrics,
        per_series=per_series,
        per_horizon=per_horizon,
        best_method=best,
        ranking_metric="validation_rmse",
        runtime=runtime,
        evaluation="rolling validation; final test must remain outside input",
        quality_flags=flags,
        packages=packages,
        n_series=int(frame.unique_id.nunique()),
    )
    for name, data, description in (
        (
            "backtest_predictions.csv",
            predictions,
            "Keyed rolling-validation predictions and training-only MASE scales.",
        ),
        (
            "forecast.csv",
            pd.concat(forecasts, ignore_index=True),
            "Future forecasts from each refitted model.",
        ),
    ):
        artifacts.append(
            write_dataframe_artifact(
                dataframe=data,
                path=output / name,
                description=description,
                created_by="forecast-panel",
            )
        )
    artifacts.append(
        write_json_artifact(
            data=summary,
            path=output / "metrics.json",
            description="Scores, options, timings and backend versions.",
            created_by="forecast-panel",
        )
    )
    report = _report(summary)
    artifacts.append(
        write_text_artifact(
            content=report,
            path=output / "report.md",
            description="Panel comparison report from saved scores.",
            created_by="forecast-panel",
        )
    )
    if not skip_plots:
        try:
            import matplotlib.pyplot as plt

            # One validation origin for up to four series; avoid unbounded figure sizes.
            sample = predictions[predictions.window == n_windows - 1]
            ids = sorted(sample.unique_id.unique())[:4]
            fig, axes = plt.subplots(
                len(ids), 1, figsize=(10, 3 * len(ids)), squeeze=False
            )
            for axis, series_id in zip(axes[:, 0], ids):
                group = sample[sample.unique_id == series_id]
                actual = group.drop_duplicates("ds").sort_values("ds")
                axis.plot(actual.ds, actual.y, label="actual", color="black")
                for method in methods:
                    line = group[group.model == method].sort_values("ds")
                    axis.plot(line.ds, line.prediction, label=method)
                axis.set_title(series_id)
                axis.set_ylabel("target")
                axis.legend()
            fig.tight_layout()
            artifacts.append(
                write_plot_artifact(
                    figure=fig,
                    path=output / "backtest.png",
                    description="Last validation origin for up to four series.",
                    created_by="forecast-panel",
                )
            )
            plt.close(fig)
        except ImportError:
            flags.append("plot_skipped")
            warnings.append("Install ts-agents[viz] to generate plots.")
            summary["quality_flags"] = flags
            # Persist final flags even when the optional plot dependency is absent.
            write_json_artifact(
                data=summary,
                path=output / "metrics.json",
                description="Panel metrics.",
                created_by="forecast-panel",
            )
    payload = ToolPayload(
        kind="workflow",
        summary=f"Panel forecast completed; best by validation RMSE: {best}.",
        status="degraded" if warnings or flags else "ok",
        data=summary,
        artifacts=artifacts,
        warnings=warnings,
        provenance=panel_input.provenance,
    )
    return attach_workflow_run_metadata(
        payload,
        workflow_name="forecast-panel",
        output_dir=output,
        run_id=run_id,
        source=panel_input.provenance.get("series_ref", {}),
        options=options,
        resumed=resumed,
        output_dir_mode=output_dir_mode,
        defer_finalization=defer_finalization,
    )


def _metrics(rows, keys):
    output = []
    for name, group in rows.groupby(keys, sort=True):
        names = name if isinstance(name, tuple) else (name,)
        error = group.prediction.to_numpy() - group.y.to_numpy()
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        if not np.isfinite([mae, rmse]).all():
            raise ValueError(
                "Error metrics overflow Float64; rescale targets before evaluation."
            )
        scales = group.mase_scale.to_numpy()
        valid = scales > 0
        output.append(
            {
                **dict(zip(keys, names)),
                "mae": mae,
                "rmse": rmse,
                "mase": float(np.mean(np.abs(error[valid]) / scales[valid]))
                if valid.any()
                else None,
                "n_predictions": len(group),
                "n_mase_predictions": int(valid.sum()),
            }
        )
    return output


def _report(summary):
    lines = [
        "# Panel forecasting report",
        "",
        f"Series: {summary['n_series']}",
        "",
        "Scores below are rolling validation scores used for model selection. Keep a final test outside the input.",
        "",
        "| Model | MAE | RMSE | MASE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["metrics"]:
        mase = f"{row['mase']:.4f}" if row["mase"] is not None else "undefined"
        lines.append(
            f"| {row['model']} | {row['mae']:.4f} | {row['rmse']:.4f} | {mase} |"
        )
    lines.extend(
        [
            "",
            f"Best by validation RMSE: {summary['best_method']}.",
            "",
            "Metrics pool forecast origins and series; inspect per-series and per-horizon scores in metrics.json.",
            "MASE uses each origin's training history. Zero scales are excluded and counted explicitly.",
            "NHITS uses a training-prefix tail for internal early stopping. Final fits retain this internal validation tail.",
            "Saved native models are under models/. Load only trusted model artifacts.",
            "See backtest_predictions.csv, forecast.csv, metrics.json and run_manifest.json for reproducibility.",
        ]
    )
    return "\n".join(lines) + "\n"
