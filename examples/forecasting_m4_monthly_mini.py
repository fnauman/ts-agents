"""Run the professional forecasting workflow on the vendored M4 Monthly mini-panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.core.forecasting import (
    forecast_arima,
    forecast_ets,
    forecast_theta,
    forecast_seasonal_naive,
)


DEFAULT_SERIES = ("M4", "M10", "M100", "M1000", "M1002")
DEFAULT_METHODS = ("seasonal_naive", "theta", "ets", "arima")
DEFAULT_HORIZON = 18
DEFAULT_SEASON_LENGTH = 12
DEFAULT_ROLLING_ORIGINS = 2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "m4_monthly_mini.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "reports" / "forecasting-workflow-m4-mini"


def _parse_csv_list(value: str | None, default: Iterable[str]) -> list[str]:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_panel(dataset_path: Path, series_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    df = pd.read_csv(dataset_path)
    required_columns = {"unique_id", "split", "ds", "y"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns in {dataset_path}: {missing_columns}"
        )
    df = df.loc[:, ["unique_id", "split", "ds", "y"]]

    available_ids = set(df["unique_id"])
    missing_ids = [series_id for series_id in series_ids if series_id not in available_ids]
    if missing_ids:
        raise ValueError(f"Unknown series IDs requested: {missing_ids}")

    panel: dict[str, dict[str, np.ndarray]] = {}
    for series_id in series_ids:
        series_df = df[df["unique_id"] == series_id].sort_values("ds")
        train_df = series_df[series_df["split"] == "train"]
        holdout_df = series_df[series_df["split"] == "holdout"]
        panel[series_id] = {
            "train": train_df["y"].to_numpy(dtype=float),
            "holdout": holdout_df["y"].to_numpy(dtype=float),
            "holdout_ds": holdout_df["ds"].to_numpy(dtype=int),
        }

    return panel


def _smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(forecast)
    contributions = np.divide(
        200.0 * np.abs(actual - forecast),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return float(np.mean(contributions))


def _mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def _rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def _score_forecast(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    return {
        "smape": _smape(actual, forecast),
        "mae": _mae(actual, forecast),
        "rmse": _rmse(actual, forecast),
    }


def _forecast_with_method(
    method: str,
    series: np.ndarray,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    model_map = {
        "seasonal_naive": forecast_seasonal_naive,
        "theta": forecast_theta,
        "ets": forecast_ets,
        "arima": forecast_arima,
    }
    if method not in model_map:
        raise ValueError(f"Unsupported method: {method}")

    result = model_map[method](
        series,
        horizon=horizon,
        season_length=season_length,
    )
    return np.asarray(result.forecast, dtype=float)


def _rolling_origins(train_length: int, horizon: int, n_origins: int) -> list[int]:
    earliest_origin = train_length - (horizon * n_origins)
    if earliest_origin <= 0:
        raise ValueError(
            "Not enough training data for the requested rolling-origin setup: "
            f"train_length={train_length}, horizon={horizon}, n_origins={n_origins}"
        )
    return [earliest_origin + (i * horizon) for i in range(n_origins)]


def _plot_holdout_forecasts(
    series_id: str,
    train: np.ndarray,
    holdout: np.ndarray,
    holdout_ds: np.ndarray,
    forecasts_by_method: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    train_ds = np.arange(1, len(train) + 1)

    ax.plot(train_ds, train, label="train", color="black", linewidth=1.8)
    ax.plot(holdout_ds, holdout, label="holdout", color="black", linestyle="--")

    for method, forecast in forecasts_by_method.items():
        ax.plot(holdout_ds, forecast, label=method)

    ax.set_title(f"{series_id}: train, holdout, and forecasts")
    ax.set_xlabel("ds")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def _write_report(
    report_path: Path,
    dataset_path: Path,
    series_ids: list[str],
    methods: list[str],
    horizon: int,
    season_length: int,
    rolling_origins: int,
    summary_df: pd.DataFrame,
    plot_paths: list[Path],
    output_dir: Path,
) -> None:
    holdout_summary = (
        summary_df[summary_df["phase"] == "holdout"]
        .sort_values(["smape", "mae", "rmse"])
        .reset_index(drop=True)
    )
    best_method = holdout_summary.iloc[0]["method"] if not holdout_summary.empty else "n/a"

    lines = [
        "# Professional Forecasting Workflow Report",
        "",
        "## Dataset",
        f"- dataset: `{dataset_path}`",
        f"- series: `{', '.join(series_ids)}`",
        f"- horizon: `{horizon}`",
        f"- season length: `{season_length}`",
        "",
        "## Protocol",
        f"- rolling-origin backtesting: `{rolling_origins}` origins per series",
        "- final evaluation: official 18-step holdout",
        f"- methods: `{', '.join(methods)}`",
        "",
        "## Holdout ranking",
        "",
    ]
    lines.extend(_markdown_table(holdout_summary, ["method", "smape", "mae", "rmse"]))
    lines.append("")
    lines.extend([
        "## Recommendation",
        f"- recommended method: `{best_method}`",
        "- ranking rule: lowest holdout sMAPE, with MAE/RMSE as tie-breakers",
        "",
        "## Generated artifacts",
        f"- metrics: `{output_dir / 'metrics_by_series.csv'}`",
        f"- summary: `{output_dir / 'summary.csv'}`",
        f"- holdout forecasts: `{output_dir / 'holdout_forecasts.csv'}`",
        f"- machine-readable summary: `{output_dir / 'run_summary.json'}`",
    ])
    lines.append("")

    if plot_paths:
        lines.extend(["", "## Plots"])
        for plot_path in plot_paths:
            relative_plot = plot_path.relative_to(output_dir)
            lines.append(f"- `{relative_plot}`")
            lines.append(f"  ![]({relative_plot.as_posix()})")

    lines.extend(
        [
            "",
            "## Limitations",
            "- This is a fixed five-series reference workflow, not the full M4 benchmark.",
            "- The first implementation focuses on deterministic, reproducible artifacts rather than leaderboard tuning.",
            "- The reduced smoke profile validates artifact production and structure, not benchmark leadership claims.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_workflow(
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    series_ids: list[str] | None = None,
    methods: list[str] | None = None,
    horizon: int = DEFAULT_HORIZON,
    season_length: int = DEFAULT_SEASON_LENGTH,
    rolling_origins: int = DEFAULT_ROLLING_ORIGINS,
    plot_series: list[str] | None = None,
) -> dict[str, object]:
    series_ids = list(series_ids) if series_ids is not None else list(DEFAULT_SERIES)
    methods = list(methods) if methods is not None else list(DEFAULT_METHODS)

    if not series_ids:
        raise ValueError("at least one series ID is required")
    if not methods:
        raise ValueError("at least one forecasting method is required")

    plot_series = list(plot_series) if plot_series is not None else series_ids[:1]
    invalid_plot_series = [series_id for series_id in plot_series if series_id not in series_ids]
    if invalid_plot_series:
        raise ValueError(f"plot series must be a subset of --series: {invalid_plot_series}")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(dataset_path, series_ids)

    metrics_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    plot_paths: list[Path] = []

    for series_id in series_ids:
        train = panel[series_id]["train"]
        full_holdout = panel[series_id]["holdout"]
        full_holdout_ds = panel[series_id]["holdout_ds"]
        if horizon > len(full_holdout):
            raise ValueError(
                f"horizon {horizon} exceeds holdout length for {series_id}: "
                f"{len(full_holdout)}"
            )
        holdout = full_holdout[:horizon]
        holdout_ds = full_holdout_ds[:horizon]

        forecasts_by_method: dict[str, np.ndarray] = {}

        for origin_index, origin_end in enumerate(
            _rolling_origins(len(train), horizon, rolling_origins),
            start=1,
        ):
            origin_train = train[:origin_end]
            origin_actual = train[origin_end: origin_end + horizon]

            for method in methods:
                origin_forecast = _forecast_with_method(
                    method,
                    origin_train,
                    horizon,
                    season_length,
                )
                origin_scores = _score_forecast(origin_actual, origin_forecast)
                metrics_rows.append(
                    {
                        "phase": "rolling_origin",
                        "origin": origin_index,
                        "unique_id": series_id,
                        "method": method,
                        **origin_scores,
                    }
                )

        for method in methods:
            holdout_forecast = _forecast_with_method(
                method,
                train,
                horizon,
                season_length,
            )
            forecasts_by_method[method] = holdout_forecast
            holdout_scores = _score_forecast(holdout, holdout_forecast)
            metrics_rows.append(
                {
                    "phase": "holdout",
                    "origin": "official",
                    "unique_id": series_id,
                    "method": method,
                    **holdout_scores,
                }
            )

            for ds_value, actual_value, forecast_value in zip(
                holdout_ds,
                holdout,
                holdout_forecast,
            ):
                forecast_rows.append(
                    {
                        "unique_id": series_id,
                        "method": method,
                        "ds": int(ds_value),
                        "actual": float(actual_value),
                        "forecast": float(forecast_value),
                    }
                )

        if series_id in plot_series:
            plot_path = plots_dir / f"{series_id}.png"
            _plot_holdout_forecasts(
                series_id=series_id,
                train=train,
                holdout=holdout,
                holdout_ds=holdout_ds,
                forecasts_by_method=forecasts_by_method,
                output_path=plot_path,
            )
            plot_paths.append(plot_path)

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = (
        metrics_df.groupby(["phase", "method"], as_index=False)[["smape", "mae", "rmse"]]
        .mean()
        .sort_values(["phase", "smape", "mae", "rmse"])
    )
    holdout_summary = summary_df[summary_df["phase"] == "holdout"].reset_index(drop=True)
    best_method = holdout_summary.iloc[0]["method"] if not holdout_summary.empty else None

    metrics_path = output_dir / "metrics_by_series.csv"
    summary_path = output_dir / "summary.csv"
    forecasts_path = output_dir / "holdout_forecasts.csv"
    report_path = output_dir / "REPORT.md"
    run_summary_path = output_dir / "run_summary.json"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(forecast_rows).to_csv(forecasts_path, index=False)

    run_summary_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "series": series_ids,
                "methods": methods,
                "horizon": horizon,
                "season_length": season_length,
                "rolling_origins": rolling_origins,
                "best_method": best_method,
                "artifacts": {
                    "metrics_by_series": str(metrics_path),
                    "summary": str(summary_path),
                    "holdout_forecasts": str(forecasts_path),
                    "report": str(report_path),
                    "plots": [str(path) for path in plot_paths],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_report(
        report_path=report_path,
        dataset_path=dataset_path,
        series_ids=series_ids,
        methods=methods,
        horizon=horizon,
        season_length=season_length,
        rolling_origins=rolling_origins,
        summary_df=summary_df,
        plot_paths=plot_paths,
        output_dir=output_dir,
    )

    return {
        "dataset": dataset_path,
        "output_dir": output_dir,
        "series": series_ids,
        "methods": methods,
        "horizon": horizon,
        "season_length": season_length,
        "rolling_origins": rolling_origins,
        "best_method": best_method,
        "artifacts": {
            "metrics_by_series": metrics_path,
            "summary": summary_path,
            "holdout_forecasts": forecasts_path,
            "report": report_path,
            "run_summary": run_summary_path,
            "plots": plot_paths,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the professional forecasting workflow on the M4 Monthly mini-panel.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the vendored M4 Monthly mini-panel CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where artifacts will be written.",
    )
    parser.add_argument(
        "--series",
        default=None,
        help="Comma-separated series IDs to evaluate. Default: all reference series.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated forecasting methods. Default: seasonal_naive,theta,ets,arima.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Forecast horizon for rolling-origin and holdout evaluation.",
    )
    parser.add_argument(
        "--season-length",
        type=int,
        default=DEFAULT_SEASON_LENGTH,
        help="Season length forwarded to the forecasting methods.",
    )
    parser.add_argument(
        "--rolling-origins",
        type=int,
        default=DEFAULT_ROLLING_ORIGINS,
        help="Number of expanding-window backtest origins per series.",
    )
    parser.add_argument(
        "--plot-series",
        default=None,
        help="Comma-separated series IDs to plot. Default: the first evaluated series.",
    )
    args = parser.parse_args()

    result = run_workflow(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        series_ids=_parse_csv_list(args.series, DEFAULT_SERIES),
        methods=_parse_csv_list(args.methods, DEFAULT_METHODS),
        horizon=args.horizon,
        season_length=args.season_length,
        rolling_origins=args.rolling_origins,
        plot_series=_parse_csv_list(args.plot_series, []) if args.plot_series is not None else None,
    )

    print(f"Wrote professional forecasting workflow artifacts to {result['output_dir']}")
    print(f"Best holdout method: {result['best_method']}")
    print(f"Summary: {result['artifacts']['summary']}")


if __name__ == "__main__":
    main()
