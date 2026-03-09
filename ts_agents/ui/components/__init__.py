"""Gradio UI components for time series analysis.

Each component creates a tab or section of the interface.
"""

from .chat import create_chat_tab
from .decomposition import create_decomposition_tab
from .forecasting import create_forecasting_tab
from .patterns import create_patterns_tab
from .classification import create_classification_tab
from .comparison import create_comparison_tab

__all__ = [
    "create_chat_tab",
    "create_decomposition_tab",
    "create_forecasting_tab",
    "create_patterns_tab",
    "create_classification_tab",
    "create_comparison_tab",
]
