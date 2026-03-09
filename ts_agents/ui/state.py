"""Session state management for the Gradio UI.

This module provides state management that integrates with the persistence layer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..persistence import SessionState, SessionStore, get_default_store


@dataclass
class UIState:
    """Enhanced state for the Gradio UI.

    This wraps SessionState and adds UI-specific state management.
    """
    # Core session state (persisted)
    session: SessionState = field(default_factory=SessionState)

    # Loaded data (not persisted, loaded on demand)
    loaded_series: Optional[Any] = None
    loaded_dataframe: Optional[Any] = None

    # Current analysis results (not persisted)
    current_decomposition: Optional[Dict[str, Any]] = None
    current_forecast: Optional[Dict[str, Any]] = None
    current_patterns: Optional[Dict[str, Any]] = None
    current_comparison: Optional[Dict[str, Any]] = None

    # Agent state
    agent_chat: Optional[Any] = None

    @property
    def run_id(self) -> Optional[str]:
        return self.session.current_run_id

    @run_id.setter
    def run_id(self, value: str):
        self.session.current_run_id = value
        self.loaded_series = None  # Clear cached series

    @property
    def variable(self) -> Optional[str]:
        return self.session.current_variable

    @variable.setter
    def variable(self, value: str):
        self.session.current_variable = value
        self.loaded_series = None  # Clear cached series

    @property
    def chat_messages(self) -> List[Dict[str, str]]:
        return self.session.chat_messages

    def add_chat_message(self, role: str, content: str):
        """Add a message to chat history."""
        self.session.add_chat_message(role, content)

    def clear_chat(self):
        """Clear chat history."""
        self.session.clear_chat()
        if self.agent_chat is not None:
            self.agent_chat.reset()

    def add_analysis(self, method: str, params: Dict[str, Any], result_summary: str = None):
        """Record an analysis in history."""
        self.session.add_analysis(method, params, result_summary)

    def get_series(self) -> Optional[Any]:
        """Get the currently selected series, loading if needed."""
        if self.loaded_series is not None:
            return self.loaded_series

        if not self.run_id or not self.variable:
            return None

        try:
            from ..data_access import get_series

            self.loaded_series = get_series(
                self.run_id,
                self.variable,
            )
            return self.loaded_series
        except Exception as e:
            print(f"Error loading series: {e}")
            return None

    def save(self, store: Optional[SessionStore] = None):
        """Save session to persistent storage."""
        if store is None:
            store = get_default_store()
        store.save(self.session)

    @classmethod
    def load(cls, session_id: str, store: Optional[SessionStore] = None) -> "UIState":
        """Load a session from storage."""
        if store is None:
            store = get_default_store()

        session = store.load(session_id)
        if session is None:
            return cls()

        return cls(session=session)

    @classmethod
    def load_latest(cls, store: Optional[SessionStore] = None) -> "UIState":
        """Load the most recent session."""
        if store is None:
            store = get_default_store()

        session = store.load_latest()
        if session is None:
            return cls()

        return cls(session=session)


def create_initial_state() -> UIState:
    """Create initial UI state, optionally restoring from persistence."""
    try:
        return UIState.load_latest()
    except Exception:
        return UIState()


def save_state_on_change(state: UIState, store: Optional[SessionStore] = None):
    """Save state when it changes."""
    try:
        state.save(store)
    except Exception as e:
        print(f"Warning: Failed to save state: {e}")


def load_series_data(run_id: str, variable: str) -> Optional[Any]:
    """Load time series data for a given run and variable.

    This is a shared utility function used by UI components to load data.
    Centralizes data loading logic to avoid duplication across components.

    Parameters
    ----------
    run_id : str
        The run identifier (e.g., "Re200Rm200")
    variable : str
        The variable name (e.g., "bx001_real")

    Returns
    -------
    np.ndarray or None
        The time series data, or None if loading failed
    """
    import numpy as np

    try:
        from ..data_access import get_series

        return get_series(run_id, variable)
    except Exception as e:
        print(f"Error loading data for {run_id}/{variable}: {e}")
        return None
