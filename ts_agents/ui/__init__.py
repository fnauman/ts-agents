"""Gradio UI for Time Series Analysis.

This module provides a comprehensive Gradio interface for:
- Agent-based analysis (chat interface)
- Manual analysis tabs (decomposition, forecasting, patterns, etc.)
- Method comparison
- Session persistence
"""

from .gradio_app import create_app, launch_app
from .state import UIState

__all__ = [
    "create_app",
    "launch_app",
    "UIState",
]
