"""Session store for Gradio UI state persistence.

This module provides session management for the Gradio interface,
allowing users to restore their analysis sessions across browser reloads.
"""

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """State of a user session in the Gradio UI.

    This captures all the user's selections and loaded data
    so it can be restored on page reload or browser restart.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    updated: str = field(default_factory=lambda: datetime.now().isoformat())

    # Data selection
    current_run_id: Optional[str] = None
    current_variable: Optional[str] = None

    # Analysis history
    analysis_history: List[Dict[str, Any]] = field(default_factory=list)

    # UI state per tab
    decomposition_state: Dict[str, Any] = field(default_factory=dict)
    forecasting_state: Dict[str, Any] = field(default_factory=dict)
    patterns_state: Dict[str, Any] = field(default_factory=dict)
    classification_state: Dict[str, Any] = field(default_factory=dict)
    spectral_state: Dict[str, Any] = field(default_factory=dict)

    # Chat history (for agent mode)
    chat_messages: List[Dict[str, str]] = field(default_factory=list)

    # User preferences
    preferences: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def add_analysis(
        self,
        method: str,
        params: Dict[str, Any],
        result_summary: Optional[str] = None,
    ) -> None:
        """Add an analysis to the history."""
        self.analysis_history.append({
            "method": method,
            "params": params,
            "result_summary": result_summary,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.current_run_id,
            "variable": self.current_variable,
        })
        self.updated = datetime.now().isoformat()

    def add_chat_message(self, role: str, content: str) -> None:
        """Add a chat message to history."""
        self.chat_messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.updated = datetime.now().isoformat()

    def clear_chat(self) -> None:
        """Clear chat history."""
        self.chat_messages = []
        self.updated = datetime.now().isoformat()


class SessionStore:
    """Persistent storage for Gradio sessions.

    Saves session state to disk so users can restore their work
    across browser reloads or even across different sessions.

    Parameters
    ----------
    root_dir : str or Path
        Directory for session files
    max_sessions : int
        Maximum number of sessions to retain
    session_timeout_hours : int
        Hours before a session is considered stale

    Examples
    --------
    >>> store = SessionStore("./sessions")
    >>> state = SessionState()
    >>> state.current_run_id = "Re200Rm200"
    >>> store.save(state)
    >>> restored = store.load(state.session_id)
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "./sessions",
        max_sessions: int = 100,
        session_timeout_hours: int = 168,  # 7 days
    ):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max_sessions
        self.session_timeout_hours = session_timeout_hours
        self._sessions_index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._load_index()

    @property
    def index_file(self) -> Path:
        """Path to the sessions index file."""
        return self.root / "sessions_index.json"

    def _load_index(self) -> None:
        """Load the sessions index from disk (thread-safe)."""
        with self._lock:
            if self.index_file.exists():
                try:
                    self._sessions_index = json.loads(self.index_file.read_text())
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load sessions index: {e}")
                    self._sessions_index = {}
            else:
                self._sessions_index = {}

    def _save_index(self) -> None:
        """Save the sessions index to disk (thread-safe)."""
        try:
            with self._lock:
                self.index_file.write_text(json.dumps(self._sessions_index, indent=2))
        except IOError as e:
            logger.error(f"Failed to save sessions index: {e}")

    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.root / f"session_{session_id}.json"

    def _is_stale(self, session_info: Dict[str, Any]) -> bool:
        """Check if a session is stale (past timeout)."""
        updated = datetime.fromisoformat(session_info.get("updated", "1970-01-01"))
        age_hours = (datetime.now() - updated).total_seconds() / 3600
        return age_hours > self.session_timeout_hours

    def save(self, state: SessionState) -> str:
        """Save a session state.

        Parameters
        ----------
        state : SessionState
            Session state to save

        Returns
        -------
        str
            Session ID
        """
        state.updated = datetime.now().isoformat()
        session_path = self._get_session_path(state.session_id)

        with self._lock:
            try:
                session_path.write_text(json.dumps(state.to_dict(), indent=2))
            except IOError as e:
                logger.error(f"Failed to save session: {e}")
                return state.session_id

            # Update index
            self._sessions_index[state.session_id] = {
                "created": state.created,
                "updated": state.updated,
                "run_id": state.current_run_id,
                "variable": state.current_variable,
                "analysis_count": len(state.analysis_history),
                "chat_count": len(state.chat_messages),
            }

        self._save_index()

        # Cleanup old sessions if needed
        self._cleanup_if_needed()

        logger.debug(f"Saved session {state.session_id}")
        return state.session_id

    def load(self, session_id: str) -> Optional[SessionState]:
        """Load a session state.

        Parameters
        ----------
        session_id : str
            Session ID to load

        Returns
        -------
        SessionState or None
            Loaded session state or None if not found
        """
        session_path = self._get_session_path(session_id)

        if not session_path.exists():
            logger.debug(f"Session not found: {session_id}")
            return None

        try:
            data = json.loads(session_path.read_text())
            state = SessionState.from_dict(data)
            logger.debug(f"Loaded session {session_id}")
            return state
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return None

    def load_latest(self) -> Optional[SessionState]:
        """Load the most recently updated session.

        Returns
        -------
        SessionState or None
            Most recent session or None if no sessions exist
        """
        if not self._sessions_index:
            return None

        # Find most recent non-stale session
        latest_id = None
        latest_time = None

        for session_id, info in self._sessions_index.items():
            if self._is_stale(info):
                continue

            updated = datetime.fromisoformat(info["updated"])
            if latest_time is None or updated > latest_time:
                latest_time = updated
                latest_id = session_id

        if latest_id:
            return self.load(latest_id)
        return None

    def delete(self, session_id: str) -> bool:
        """Delete a session.

        Parameters
        ----------
        session_id : str
            Session ID to delete

        Returns
        -------
        bool
            True if deleted, False if not found
        """
        session_path = self._get_session_path(session_id)

        if session_path.exists():
            session_path.unlink()

        if session_id in self._sessions_index:
            del self._sessions_index[session_id]
            self._save_index()
            logger.debug(f"Deleted session {session_id}")
            return True

        return False

    def list_sessions(
        self,
        include_stale: bool = False,
    ) -> List[Dict[str, Any]]:
        """List all sessions.

        Parameters
        ----------
        include_stale : bool
            Include sessions past timeout

        Returns
        -------
        list of dict
            Session metadata sorted by update time (newest first)
        """
        sessions = []

        for session_id, info in self._sessions_index.items():
            if not include_stale and self._is_stale(info):
                continue

            sessions.append({
                "session_id": session_id,
                **info,
            })

        # Sort by update time, newest first
        sessions.sort(
            key=lambda s: s.get("updated", ""),
            reverse=True
        )

        return sessions

    def _cleanup_if_needed(self) -> None:
        """Remove old sessions if over limit."""
        # First remove stale sessions
        stale_ids = [
            sid for sid, info in self._sessions_index.items()
            if self._is_stale(info)
        ]

        for sid in stale_ids:
            self.delete(sid)

        # If still over limit, remove oldest
        if len(self._sessions_index) > self.max_sessions:
            sessions = list(self._sessions_index.items())
            sessions.sort(key=lambda x: x[1].get("updated", ""))

            to_remove = len(sessions) - self.max_sessions
            for sid, _ in sessions[:to_remove]:
                self.delete(sid)

    def cleanup_stale(self) -> int:
        """Remove all stale sessions.

        Returns
        -------
        int
            Number of sessions removed
        """
        stale_ids = [
            sid for sid, info in self._sessions_index.items()
            if self._is_stale(info)
        ]

        for sid in stale_ids:
            self.delete(sid)

        return len(stale_ids)

    def stats(self) -> Dict[str, Any]:
        """Get session store statistics.

        Returns
        -------
        dict
            Statistics about stored sessions
        """
        active = 0
        stale = 0

        for info in self._sessions_index.values():
            if self._is_stale(info):
                stale += 1
            else:
                active += 1

        return {
            "total_sessions": len(self._sessions_index),
            "active_sessions": active,
            "stale_sessions": stale,
            "max_sessions": self.max_sessions,
            "timeout_hours": self.session_timeout_hours,
            "root_dir": str(self.root),
        }


# Default session store instance
_default_store: Optional[SessionStore] = None


def get_default_store() -> SessionStore:
    """Get the default session store, creating it if needed."""
    global _default_store
    if _default_store is None:
        _default_store = SessionStore()
    return _default_store


def set_default_store(store: SessionStore) -> None:
    """Set the default session store instance."""
    global _default_store
    _default_store = store


def init_session_store(
    root_dir: Union[str, Path] = "./sessions",
    max_sessions: int = 100,
    session_timeout_hours: int = 168,
) -> SessionStore:
    """Initialize and set the default session store.

    Parameters
    ----------
    root_dir : str or Path
        Directory for session files
    max_sessions : int
        Maximum sessions to retain
    session_timeout_hours : int
        Hours before session is stale

    Returns
    -------
    SessionStore
        The initialized store instance
    """
    store = SessionStore(root_dir, max_sessions, session_timeout_hours)
    set_default_store(store)
    return store
