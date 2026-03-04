"""Tests for the session store module."""

import tempfile
import shutil
from pathlib import Path

import pytest

from src.persistence.session_store import (
    SessionState,
    SessionStore,
    init_session_store,
    get_default_store,
    set_default_store,
)


class TestSessionState:
    """Tests for SessionState class."""

    def test_session_state_creation(self):
        """Test creating a session state."""
        state = SessionState()

        assert state.session_id is not None
        assert len(state.session_id) == 8
        assert state.created is not None
        assert state.current_run_id is None
        assert state.analysis_history == []

    def test_session_state_to_dict(self):
        """Test converting session state to dict."""
        state = SessionState()
        state.current_run_id = "Re200Rm200"
        state.current_variable = "bx001_real"

        data = state.to_dict()

        assert data["current_run_id"] == "Re200Rm200"
        assert data["current_variable"] == "bx001_real"
        assert "session_id" in data

    def test_session_state_from_dict(self):
        """Test creating session state from dict."""
        data = {
            "session_id": "abc12345",
            "created": "2024-01-01T00:00:00",
            "current_run_id": "Re200Rm200",
            "current_variable": "bx001_real",
            "analysis_history": [],
        }

        state = SessionState.from_dict(data)

        assert state.session_id == "abc12345"
        assert state.current_run_id == "Re200Rm200"

    def test_add_analysis(self):
        """Test adding analysis to history."""
        state = SessionState()
        state.current_run_id = "Re200Rm200"
        state.current_variable = "bx001_real"

        state.add_analysis(
            method="stl_decompose",
            params={"period": 150},
            result_summary="Decomposition completed",
        )

        assert len(state.analysis_history) == 1
        assert state.analysis_history[0]["method"] == "stl_decompose"
        assert state.analysis_history[0]["run_id"] == "Re200Rm200"

    def test_add_chat_message(self):
        """Test adding chat messages."""
        state = SessionState()

        state.add_chat_message("user", "Hello")
        state.add_chat_message("assistant", "Hi there!")

        assert len(state.chat_messages) == 2
        assert state.chat_messages[0]["role"] == "user"
        assert state.chat_messages[1]["content"] == "Hi there!"

    def test_clear_chat(self):
        """Test clearing chat history."""
        state = SessionState()
        state.add_chat_message("user", "Hello")
        state.add_chat_message("assistant", "Hi")

        state.clear_chat()

        assert state.chat_messages == []


class TestSessionStore:
    """Tests for SessionStore class."""

    @pytest.fixture
    def temp_store_dir(self):
        """Create a temporary directory for store testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def store(self, temp_store_dir):
        """Create a store instance for testing."""
        return SessionStore(root_dir=temp_store_dir)

    def test_store_initialization(self, temp_store_dir):
        """Test store initializes correctly."""
        store = SessionStore(root_dir=temp_store_dir)

        assert store.root.exists()
        # Index file is created on first save, not on initialization
        assert store._sessions_index == {}

    def test_save_and_load(self, store):
        """Test saving and loading a session."""
        state = SessionState()
        state.current_run_id = "Re200Rm200"
        state.current_variable = "bx001_real"

        session_id = store.save(state)

        loaded = store.load(session_id)

        assert loaded is not None
        assert loaded.session_id == state.session_id
        assert loaded.current_run_id == "Re200Rm200"
        assert loaded.current_variable == "bx001_real"

    def test_load_nonexistent(self, store):
        """Test loading a non-existent session."""
        loaded = store.load("nonexistent")
        assert loaded is None

    def test_load_latest(self, store):
        """Test loading the most recent session."""
        state1 = SessionState()
        state1.current_run_id = "run1"
        store.save(state1)

        state2 = SessionState()
        state2.current_run_id = "run2"
        store.save(state2)

        latest = store.load_latest()

        assert latest is not None
        assert latest.current_run_id == "run2"

    def test_delete_session(self, store):
        """Test deleting a session."""
        state = SessionState()
        store.save(state)

        result = store.delete(state.session_id)
        assert result is True

        loaded = store.load(state.session_id)
        assert loaded is None

    def test_delete_nonexistent(self, store):
        """Test deleting a non-existent session."""
        result = store.delete("nonexistent")
        assert result is False

    def test_list_sessions(self, store):
        """Test listing sessions."""
        state1 = SessionState()
        state1.current_run_id = "run1"
        store.save(state1)

        state2 = SessionState()
        state2.current_run_id = "run2"
        store.save(state2)

        sessions = store.list_sessions()

        assert len(sessions) == 2
        # Should be sorted by update time, newest first
        assert sessions[0]["run_id"] == "run2"

    def test_max_sessions_cleanup(self, temp_store_dir):
        """Test that old sessions are cleaned up when over limit."""
        store = SessionStore(root_dir=temp_store_dir, max_sessions=2)

        for i in range(5):
            state = SessionState()
            state.current_run_id = f"run{i}"
            store.save(state)

        sessions = store.list_sessions()
        assert len(sessions) <= 2

    def test_stats(self, store):
        """Test getting store statistics."""
        state = SessionState()
        store.save(state)

        stats = store.stats()

        assert stats["total_sessions"] == 1
        assert stats["active_sessions"] == 1
        assert stats["stale_sessions"] == 0


class TestDefaultStore:
    """Tests for default store management."""

    @pytest.fixture
    def temp_store_dir(self):
        """Create a temporary directory for store testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        set_default_store(None)

    def test_init_session_store(self, temp_store_dir):
        """Test init_session_store sets the default store."""
        store = init_session_store(root_dir=temp_store_dir)

        default = get_default_store()
        assert default is store
