"""Time series decomposition methods.

This module provides functions for decomposing time series into
trend, seasonal, and residual components using various methods:

- STL (Seasonal-Trend LOESS): Robust decomposition for single seasonality
- MSTL: Multi-Seasonal STL for multiple periodicities
- Holt-Winters: Exponential smoothing decomposition
"""

from .stl import stl_decompose
from .mstl import mstl_decompose
from .holt_winters import holt_winters_decompose

__all__ = [
    "stl_decompose",
    "mstl_decompose",
    "holt_winters_decompose",
]
