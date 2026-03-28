"""Statistical forecasting methods for time series.

This module provides functions for time series forecasting using statistical
models from the statsforecast library.
"""

from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd

from ..base import ForecastResult, MultiForecastResult

StatsForecast = None
AutoARIMA = None
AutoETS = None
AutoTheta = None
SeasonalNaive = None


def _get_statsforecast_components():
    global StatsForecast, AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

    if all(
        component is not None
        for component in (StatsForecast, AutoARIMA, AutoETS, AutoTheta, SeasonalNaive)
    ):
        return StatsForecast, AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

    try:
        from statsforecast import StatsForecast as imported_statsforecast
        from statsforecast.models import (
            AutoARIMA as imported_auto_arima,
            AutoETS as imported_auto_ets,
            AutoTheta as imported_auto_theta,
            SeasonalNaive as imported_seasonal_naive,
        )
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Statistical forecasting requires optional dependencies. Install with: pip install "ts-agents[forecasting]"'
        ) from exc

    if StatsForecast is None:
        StatsForecast = imported_statsforecast
    if AutoARIMA is None:
        AutoARIMA = imported_auto_arima
    if AutoETS is None:
        AutoETS = imported_auto_ets
    if AutoTheta is None:
        AutoTheta = imported_auto_theta
    if SeasonalNaive is None:
        SeasonalNaive = imported_seasonal_naive

    return StatsForecast, AutoARIMA, AutoETS, AutoTheta, SeasonalNaive


def _normalize_season_length(season_length: Optional[int]) -> Optional[int]:
    """Normalize NumPy integer scalars to built-in ints for StatsForecast."""
    if season_length is None:
        return None
    if isinstance(season_length, np.integer):
        season_length = int(season_length)
    return season_length if season_length else None


def _resolve_seasonal_naive_length(season_length: Optional[int]) -> int:
    """Return a valid season length for the seasonal naive baseline."""
    normalized = _normalize_season_length(season_length)
    return normalized if normalized is not None else 1


def forecast_arima(
    series: np.ndarray,
    horizon: int = 10,
    level: List[int] = None,
    season_length: Optional[int] = None,
) -> ForecastResult:
    """Forecast using AutoARIMA.

    AutoARIMA automatically selects the best ARIMA(p,d,q) parameters.

    Parameters
    ----------
    series : np.ndarray
        1D time series data (historical values)
    horizon : int
        Number of steps to forecast
    level : list of int, optional
        Confidence levels for prediction intervals (e.g., [80, 95])

    Returns
    -------
    ForecastResult
        Forecast values and optional prediction intervals

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 10*np.pi, 200)) + 0.1 * np.random.randn(200)
    >>> result = forecast_arima(x, horizon=20)
    >>> print(f"Forecast shape: {result.forecast.shape}")
    """
    StatsForecast, AutoARIMA, _, _, _ = _get_statsforecast_components()
    series = np.asarray(series, dtype=np.float64).flatten()

    df = pd.DataFrame({
        'unique_id': 'series',
        'ds': np.arange(len(series)),
        'y': series,
    })

    normalized_season_length = _normalize_season_length(season_length)
    models = [AutoARIMA(season_length=normalized_season_length) if normalized_season_length else AutoARIMA()]
    if level is None:
        level = [95]

    sf = StatsForecast(models=models, freq=1, n_jobs=1)
    sf.fit(df)

    # Get prediction intervals if requested
    forecast_df = sf.predict(h=horizon, level=level)

    forecast_values = forecast_df['AutoARIMA'].values

    # Extract prediction intervals
    lower_bound = None
    upper_bound = None
    if level:
        lo_col = f'AutoARIMA-lo-{level[0]}'
        hi_col = f'AutoARIMA-hi-{level[0]}'
        if lo_col in forecast_df.columns:
            lower_bound = forecast_df[lo_col].values
            upper_bound = forecast_df[hi_col].values

    return ForecastResult(
        method="auto_arima",
        forecast=forecast_values,
        horizon=horizon,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=level[0] / 100 if level else 0.95,
    )


def forecast_ets(
    series: np.ndarray,
    horizon: int = 10,
    level: List[int] = None,
    season_length: Optional[int] = None,
) -> ForecastResult:
    """Forecast using AutoETS (Exponential Smoothing).

    AutoETS automatically selects the best ETS model parameters.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    horizon : int
        Number of steps to forecast
    level : list of int, optional
        Confidence levels for prediction intervals

    Returns
    -------
    ForecastResult
        Forecast values and optional prediction intervals
    """
    StatsForecast, _, AutoETS, _, _ = _get_statsforecast_components()
    series = np.asarray(series, dtype=np.float64).flatten()

    df = pd.DataFrame({
        'unique_id': 'series',
        'ds': np.arange(len(series)),
        'y': series,
    })

    normalized_season_length = _normalize_season_length(season_length)
    models = [AutoETS(season_length=normalized_season_length) if normalized_season_length else AutoETS()]
    if level is None:
        level = [95]

    sf = StatsForecast(models=models, freq=1, n_jobs=1)
    sf.fit(df)

    forecast_df = sf.predict(h=horizon, level=level)
    forecast_values = forecast_df['AutoETS'].values

    lower_bound = None
    upper_bound = None
    if level:
        lo_col = f'AutoETS-lo-{level[0]}'
        hi_col = f'AutoETS-hi-{level[0]}'
        if lo_col in forecast_df.columns:
            lower_bound = forecast_df[lo_col].values
            upper_bound = forecast_df[hi_col].values

    return ForecastResult(
        method="auto_ets",
        forecast=forecast_values,
        horizon=horizon,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=level[0] / 100 if level else 0.95,
    )


def forecast_theta(
    series: np.ndarray,
    horizon: int = 10,
    level: List[int] = None,
    season_length: Optional[int] = None,
) -> ForecastResult:
    """Forecast using AutoTheta.

    The Theta method decomposes a series into trend and seasonal components.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    horizon : int
        Number of steps to forecast
    level : list of int, optional
        Confidence levels for prediction intervals

    Returns
    -------
    ForecastResult
        Forecast values and optional prediction intervals
    """
    StatsForecast, _, _, AutoTheta, _ = _get_statsforecast_components()
    series = np.asarray(series, dtype=np.float64).flatten()

    df = pd.DataFrame({
        'unique_id': 'series',
        'ds': np.arange(len(series)),
        'y': series,
    })

    normalized_season_length = _normalize_season_length(season_length)
    models = [AutoTheta(season_length=normalized_season_length) if normalized_season_length else AutoTheta()]
    if level is None:
        level = [95]

    sf = StatsForecast(models=models, freq=1, n_jobs=1)
    sf.fit(df)

    forecast_df = sf.predict(h=horizon, level=level)
    forecast_values = forecast_df['AutoTheta'].values

    lower_bound = None
    upper_bound = None
    if level:
        lo_col = f'AutoTheta-lo-{level[0]}'
        hi_col = f'AutoTheta-hi-{level[0]}'
        if lo_col in forecast_df.columns:
            lower_bound = forecast_df[lo_col].values
            upper_bound = forecast_df[hi_col].values

    return ForecastResult(
        method="auto_theta",
        forecast=forecast_values,
        horizon=horizon,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=level[0] / 100 if level else 0.95,
    )


def forecast_seasonal_naive(
    series: np.ndarray,
    horizon: int = 10,
    level: List[int] = None,
    season_length: Optional[int] = None,
) -> ForecastResult:
    """Forecast by repeating the last observed seasonal cycle.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    horizon : int
        Number of steps to forecast
    level : list of int, optional
        Confidence levels for prediction intervals
    season_length : int, optional
        Seasonal period to repeat. Defaults to 1 when omitted.

    Returns
    -------
    ForecastResult
        Forecast values and optional prediction intervals
    """
    StatsForecast, _, _, _, SeasonalNaive = _get_statsforecast_components()
    series = np.asarray(series, dtype=np.float64).flatten()

    df = pd.DataFrame({
        'unique_id': 'series',
        'ds': np.arange(len(series)),
        'y': series,
    })

    resolved_season_length = _resolve_seasonal_naive_length(season_length)
    models = [SeasonalNaive(season_length=resolved_season_length)]
    if level is None:
        level = [95]

    sf = StatsForecast(models=models, freq=1, n_jobs=1)
    sf.fit(df)

    forecast_df = sf.predict(h=horizon, level=level)
    forecast_values = forecast_df['SeasonalNaive'].values

    lower_bound = None
    upper_bound = None
    if level:
        lo_col = f'SeasonalNaive-lo-{level[0]}'
        hi_col = f'SeasonalNaive-hi-{level[0]}'
        if lo_col in forecast_df.columns:
            lower_bound = forecast_df[lo_col].values
            upper_bound = forecast_df[hi_col].values

    return ForecastResult(
        method="seasonal_naive",
        forecast=forecast_values,
        horizon=horizon,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=level[0] / 100 if level else 0.95,
    )


def forecast_ensemble(
    series: np.ndarray,
    horizon: int = 10,
    models: List[str] = None,
    season_length: Optional[int] = None,
) -> MultiForecastResult:
    """Forecast using multiple models.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    horizon : int
        Number of steps to forecast
    models : list of str, optional
        Models to use. Default: ['arima', 'ets']
        Options: 'arima', 'ets', 'theta', 'seasonal_naive'

    Returns
    -------
    MultiForecastResult
        Forecasts from all specified models

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 10*np.pi, 200)) + np.random.randn(200) * 0.1
    >>> result = forecast_ensemble(x, horizon=20, models=['arima', 'ets', 'theta'])
    >>> ensemble = result.get_ensemble()
    >>> print(f"Ensemble forecast: {ensemble[:5]}")
    """
    StatsForecast, AutoARIMA, AutoETS, AutoTheta, SeasonalNaive = _get_statsforecast_components()
    if models is None:
        models = ['arima', 'ets']

    series = np.asarray(series, dtype=np.float64).flatten()

    df = pd.DataFrame({
        'unique_id': 'series',
        'ds': np.arange(len(series)),
        'y': series,
    })

    normalized_season_length = _normalize_season_length(season_length)
    seasonal_naive_length = _resolve_seasonal_naive_length(season_length)
    if normalized_season_length:
        model_map = {
            'arima': AutoARIMA(season_length=normalized_season_length),
            'ets': AutoETS(season_length=normalized_season_length),
            'theta': AutoTheta(season_length=normalized_season_length),
            'seasonal_naive': SeasonalNaive(season_length=seasonal_naive_length),
        }
    else:
        model_map = {
            'arima': AutoARIMA(),
            'ets': AutoETS(),
            'theta': AutoTheta(),
            'seasonal_naive': SeasonalNaive(season_length=seasonal_naive_length),
        }

    sf_models = [model_map[m] for m in models if m in model_map]

    sf = StatsForecast(models=sf_models, freq=1, n_jobs=1)
    sf.fit(df)

    forecast_df = sf.predict(h=horizon)

    forecasts = {}
    name_map = {
        'arima': 'AutoARIMA',
        'ets': 'AutoETS',
        'theta': 'AutoTheta',
        'seasonal_naive': 'SeasonalNaive',
    }

    for m in models:
        col_name = name_map.get(m)
        if col_name and col_name in forecast_df.columns:
            forecasts[m] = forecast_df[col_name].values

    return MultiForecastResult(
        method="ensemble",
        forecasts=forecasts,
        horizon=horizon,
    )


def compare_forecasts(
    series: np.ndarray,
    horizon: int = 10,
    test_size: Optional[int] = None,
    models: List[str] = None,
    season_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare forecast accuracy of multiple models using holdout validation.

    Parameters
    ----------
    series : np.ndarray
        1D time series data
    horizon : int
        Forecast horizon
    test_size : int, optional
        Number of points to use for testing. Default: horizon.
    models : list of str, optional
        Models to compare. Default: ['arima', 'ets', 'theta']
    season_length : int, optional
        Seasonal period forwarded to methods that support it.

    Returns
    -------
    dict
        Comparison results with MAE, RMSE, MAPE for each model

    Examples
    --------
    >>> x = np.sin(np.linspace(0, 20*np.pi, 500)) + 0.1 * np.random.randn(500)
    >>> results = compare_forecasts(x, horizon=20, test_size=50)
    >>> for model, metrics in results['metrics'].items():
    ...     print(f"{model}: MAE={metrics['mae']:.4f}")
    """
    if models is None:
        models = ['arima', 'ets', 'theta']

    if test_size is None:
        test_size = horizon

    series = np.asarray(series, dtype=np.float64).flatten()

    # Split into train and test
    train = series[:-test_size]
    test = series[-test_size:]

    # Get forecasts from ensemble
    result = forecast_ensemble(
        train,
        horizon=test_size,
        models=models,
        season_length=season_length,
    )

    metrics = {}
    for model, forecast in result.forecasts.items():
        # Compute error metrics
        errors = test[:len(forecast)] - forecast[:len(test)]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # MAPE (avoid division by zero)
        nonzero_mask = test[:len(forecast)] != 0
        if np.any(nonzero_mask):
            mape = float(np.mean(np.abs(errors[nonzero_mask] / test[:len(forecast)][nonzero_mask]))) * 100
        else:
            mape = np.inf

        metrics[model] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'forecast': forecast,
        }

    # Rank models by MAE
    rankings = sorted(metrics.keys(), key=lambda m: metrics[m]['mae'])

    return {
        'metrics': metrics,
        'rankings': rankings,
        'best_model': rankings[0] if rankings else None,
        'test_values': test,
    }
