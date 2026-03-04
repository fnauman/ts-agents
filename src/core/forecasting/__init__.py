"""Time series forecasting methods.

This module provides functions for:
- Statistical forecasting (ARIMA, ETS, Theta)
- Ensemble forecasting
- Forecast comparison and evaluation
"""

from .statistical import (
    forecast_arima,
    forecast_ets,
    forecast_theta,
    forecast_ensemble,
    compare_forecasts,
)

__all__ = [
    "forecast_arima",
    "forecast_ets",
    "forecast_theta",
    "forecast_ensemble",
    "compare_forecasts",
]
