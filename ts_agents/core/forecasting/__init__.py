"""Time series forecasting methods."""

from __future__ import annotations

from ts_agents._lazy import load_export

_LAZY_EXPORTS = {
    "forecast_arima": ("statistical", "forecast_arima"),
    "forecast_ets": ("statistical", "forecast_ets"),
    "forecast_theta": ("statistical", "forecast_theta"),
    "forecast_seasonal_naive": ("statistical", "forecast_seasonal_naive"),
    "forecast_ensemble": ("statistical", "forecast_ensemble"),
    "compare_forecasts": ("statistical", "compare_forecasts"),
}


def __getattr__(name: str):
    value = load_export(__name__, _LAZY_EXPORTS, name)
    globals()[name] = value
    return value

__all__ = [
    "forecast_arima",
    "forecast_ets",
    "forecast_theta",
    "forecast_seasonal_naive",
    "forecast_ensemble",
    "compare_forecasts",
]
