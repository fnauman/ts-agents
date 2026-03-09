"""Gradio UI for Time Series Analysis.

This module provides a comprehensive Gradio interface for:
- Agent-based analysis (chat interface)
- Manual analysis tabs (decomposition, forecasting, patterns, etc.)
- Method comparison
- Session persistence
"""

from typing import TYPE_CHECKING, Any

from .state import UIState

if TYPE_CHECKING:
    import gradio as gr


def create_app(*args: Any, **kwargs: Any) -> "gr.Blocks":
    from .gradio_app import create_app as _create_app

    return _create_app(*args, **kwargs)


def launch_app(*args: Any, **kwargs: Any) -> None:
    from .gradio_app import launch_app as _launch_app

    _launch_app(*args, **kwargs)

__all__ = [
    "create_app",
    "launch_app",
    "UIState",
]
