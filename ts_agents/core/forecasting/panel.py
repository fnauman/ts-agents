"""Optional Nixtla panel backends; importing this module never imports trainers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METHOD_DEPENDENCIES = {
    "seasonal_naive": (),
    "lightgbm": ("mlforecast", "lightgbm"),
    "histgbm": ("mlforecast", "sklearn"),
    "nhits": ("neuralforecast", "torch"),
}


def normalize_panel(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Validate a regular, aligned panel. No imputation or silent covariate loss."""
    required = {"unique_id", "ds", "y"}
    if set(frame.columns) != required:
        raise ValueError(
            "Panel input must contain exactly unique_id, ds, y; covariates are not supported yet."
        )
    if frame.empty or frame.isna().any().any():
        raise ValueError("Panel input must be nonempty and contain no missing values.")
    result = frame.copy()
    original_ids = result.unique_id.nunique()
    result["unique_id"] = result.unique_id.astype(str)
    if (
        result.unique_id.nunique() != original_ids
        or (result.unique_id.str.strip() == "").any()
    ):
        raise ValueError(
            "Panel IDs must be nonempty and unambiguous when converted to strings."
        )
    try:
        result["ds"] = pd.to_datetime(result.ds, errors="raise")
        result["y"] = pd.to_numeric(result.y, errors="raise").astype(float)
        offset = pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid panel dates, targets, or frequency: {exc}") from exc
    if not hasattr(result.ds, "dt") or result.ds.dt.tz is not None:
        raise ValueError(
            "Panel timestamps must use one explicit timezone-naive time convention."
        )
    if offset.n <= 0 or not np.isfinite(result.y).all():
        raise ValueError("Frequency must advance time and targets must be finite.")
    if result.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Duplicate (unique_id, ds) observations are not allowed.")
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    ends = []
    for _, group in result.groupby("unique_id", sort=False):
        expected = pd.date_range(group.ds.iloc[0], group.ds.iloc[-1], freq=offset)
        if not pd.DatetimeIndex(group.ds).equals(expected):
            raise ValueError(
                "Each series must be regular at --freq; fill or remove gaps explicitly first."
            )
        ends.append(group.ds.iloc[-1])
    if len(set(ends)) != 1:
        raise ValueError(
            "All series must end at the same timestamp for aligned panel backtests."
        )
    return result


@dataclass
class PanelBackend:
    """One fitted backend with native persistence and stable prediction columns."""

    method: str
    freq: str
    season_length: int
    config: dict[str, Any]
    estimator: Any = None
    history: pd.DataFrame | None = None

    def fit(self, frame: pd.DataFrame, horizon: int) -> None:
        if self.method == "seasonal_naive":
            self.history = frame.copy()
            return
        if self.method in {"lightgbm", "histgbm"}:
            try:
                from mlforecast import MLForecast
                from mlforecast.lag_transforms import RollingMean

                if self.method == "lightgbm":
                    from lightgbm import LGBMRegressor

                    model = LGBMRegressor(
                        n_estimators=self.config["n_estimators"],
                        random_state=self.config["seed"],
                        n_jobs=self.config["num_threads"],
                        verbosity=-1,
                    )
                else:
                    from sklearn.ensemble import HistGradientBoostingRegressor

                    model = HistGradientBoostingRegressor(
                        max_iter=self.config["n_estimators"],
                        random_state=self.config["seed"],
                        early_stopping=False,
                    )
            except ImportError as exc:
                raise ImportError(
                    "Install ts-agents[ml] for MLForecast/GBM panel methods."
                ) from exc
            self.estimator = MLForecast(
                models={self.method: model},
                freq=self.freq,
                lags=self.config["lags"],
                lag_transforms={1: [RollingMean(window_size=self.season_length)]},
                date_features=["hour", "dayofweek", "month"],
                num_threads=self.config["num_threads"],
            )
            # Input includes targets only; calendar features are known in advance.
            self.estimator.fit(frame, static_features=[])
            return
        if self.method == "nhits":
            try:
                from neuralforecast import NeuralForecast
                from neuralforecast.models import NHITS
            except ImportError as exc:
                raise ImportError(
                    "Install ts-agents[neural] for NeuralForecast/NHITS panel forecasts."
                ) from exc
            steps = self.config["max_steps"]
            model = NHITS(
                h=horizon,
                input_size=self.config["input_size"],
                max_steps=steps,
                random_seed=self.config["seed"],
                scaler_type="standard",
                val_check_steps=min(50, steps),
                early_stop_patience_steps=3,
                accelerator=self.config["accelerator"],
                devices=1,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
                alias="nhits",
            )
            self.estimator = NeuralForecast(models=[model], freq=self.freq)
            # Internal early-stopping validation is taken ONLY from this training prefix.
            self.estimator.fit(df=frame, val_size=horizon)
            return
        raise ValueError(f"Unsupported panel method: {self.method}")

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.method == "seasonal_naive":
            if self.history is None:
                raise ValueError("Fit the seasonal baseline before predicting.")
            rows: list[dict[str, Any]] = []
            for series_id, group in self.history.groupby("unique_id", sort=False):
                last = group.y.to_numpy()[-self.season_length :]
                times = pd.date_range(
                    group.ds.iloc[-1], periods=horizon + 1, freq=self.freq
                )[1:]
                rows.extend(
                    {
                        "unique_id": series_id,
                        "ds": ds,
                        "seasonal_naive": float(last[i % len(last)]),
                    }
                    for i, ds in enumerate(times)
                )
            return pd.DataFrame(rows)
        if self.method == "nhits":
            return self.estimator.predict().reset_index(drop=True)
        return self.estimator.predict(horizon)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self.method == "seasonal_naive":
            if self.history is None:
                raise ValueError("Fit the seasonal baseline before saving.")
            self.history.to_csv(directory / "history.csv", index=False)
        elif self.method == "nhits":
            self.estimator.save(path=str(directory), overwrite=False, save_dataset=True)
        else:
            self.estimator.save(str(directory))


def score_predictions(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
    method: str,
    training: pd.DataFrame,
    season_length: int,
) -> pd.DataFrame:
    """Enforce full keyed coverage and compute per-series metrics with train-only MASE."""
    columns = ["unique_id", "ds", method]
    if any(column not in predictions for column in columns):
        raise ValueError(f"{method} did not return keyed predictions.")
    predictions = predictions[columns].copy()
    if predictions.duplicated(["unique_id", "ds"]).any():
        raise ValueError(f"{method} returned duplicate predictions.")
    joined = actual.merge(
        predictions,
        on=["unique_id", "ds"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if (joined._merge != "both").any() or not np.isfinite(joined[method]).all():
        raise ValueError(
            f"{method} predictions must cover every target exactly once with finite values."
        )
    joined = joined.drop(columns="_merge")
    if not np.isfinite(joined[method].to_numpy() - joined.y.to_numpy()).all():
        raise ValueError(
            "Forecast errors overflow Float64; rescale targets before evaluation."
        )
    scales = {}
    for series_id, group in training.groupby("unique_id", sort=False):
        values = group.y.to_numpy()
        scales[series_id] = float(
            np.mean(np.abs(values[season_length:] - values[:-season_length]))
        )
    if not all(np.isfinite(scale) for scale in scales.values()):
        raise ValueError(
            "Seasonal scales overflow Float64; rescale targets before evaluation."
        )
    joined["mase_scale"] = joined.unique_id.map(scales)
    joined["model"] = method
    return joined.rename(columns={method: "prediction"})
