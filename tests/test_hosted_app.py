import importlib
import sys

import pytest


def test_hosted_app_lazily_builds_blocks(monkeypatch):
    for name in [
        "TS_AGENTS_ENABLE_AGENT",
        "TS_AGENTS_AGENT_TYPE",
        "TS_AGENTS_PERSIST_SESSIONS",
        "TS_AGENTS_UI_TITLE",
        "GRADIO_SHARE",
        "HOST",
        "PORT",
    ]:
        monkeypatch.delenv(name, raising=False)

    sys.modules.pop("ts_agents.hosted_app", None)
    module = importlib.import_module("ts_agents.hosted_app")
    captured: dict[str, object] = {}

    def fake_create_app(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "create_app", fake_create_app)

    assert module._app is None

    app = module.get_app()

    assert app is module._app
    assert module._app is app
    assert module.app is app
    assert captured == {
        "enable_agent": False,
        "agent_type": "simple",
        "persist_sessions": False,
        "title": "ts-agents Demo",
    }


def test_hosted_app_reports_missing_gradio(monkeypatch):
    sys.modules.pop("ts_agents.hosted_app", None)
    module = importlib.import_module("ts_agents.hosted_app")

    def fail_create_app(**_kwargs):
        raise ModuleNotFoundError("No module named 'gradio'", name="gradio")

    monkeypatch.setattr(module, "create_app", fail_create_app)

    with pytest.raises(ImportError, match=r'ts-agents\[ui\]'):
        module.get_app()


def test_ui_visitors_start_fresh_without_reading_shared_history(monkeypatch, tmp_path):
    from pathlib import Path
    from unittest.mock import MagicMock, Mock

    from ts_agents.persistence import SessionState, SessionStore
    from ts_agents.ui import state as state_module

    store = SessionStore(tmp_path)
    previous = SessionState()
    previous.add_chat_message("user", "another visitor's private message")
    store.save(previous)
    store.load_latest = Mock(wraps=store.load_latest)
    store.list_sessions = Mock(side_effect=AssertionError("global session listing"))
    get_store = Mock(return_value=store)
    monkeypatch.setattr(state_module, "get_default_store", get_store)
    monkeypatch.setattr("ts_agents.config.init_persistence", Mock())

    # Exercise app wiring without installing optional Gradio or UI dependencies.
    gradio = MagicMock()
    monkeypatch.setitem(sys.modules, "gradio", gradio)
    monkeypatch.setitem(sys.modules, "ts_agents.ui.components", MagicMock())
    spec = importlib.util.spec_from_file_location(
        "ts_agents.ui.gradio_app",
        Path(__file__).parents[1] / "ts_agents/ui/gradio_app.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_default_store", get_store)

    for persist in (False, True):
        gradio.reset_mock()
        get_store.reset_mock()
        module.create_app(enable_agent=False, persist_sessions=persist)
        factory = gradio.State.call_args.kwargs["value"]
        assert callable(factory)
        first, second = factory(), factory()
        assert len({previous.session_id, first.session.session_id, second.session.session_id}) == 3
        first.add_chat_message("user", "first visitor only")
        assert second.chat_messages == []
        assert first.chat_messages[0]["content"] == "first visitor only"
        assert get_store.call_count == int(persist)
        store.load_latest.assert_not_called()
        gradio.Blocks.return_value.__enter__.return_value.load.assert_not_called()
        assert gradio.State.return_value.change.call_count == int(persist)
        refresh = next(
            call.args[0] for call in gradio.Button.return_value.click.call_args_list
            if call.args[0].__name__ == "refresh_session_info"
        )
        info, history = refresh(second)
        assert previous.session_id not in info and history == []
        store.list_sessions.assert_not_called()

    # Trusted local callers retain the explicit restore API.
    restored = state_module.UIState.load(previous.session_id, store=store)
    assert restored.chat_messages[0]["content"] == "another visitor's private message"

    # The comparison callback must preserve undefined secondary metrics instead
    # of crashing when a valid zero-target forecast produces MAPE=None.
    import numpy as np
    from ts_agents.core import comparison as comparison_module

    series = np.zeros(12)
    comparison = comparison_module.compare_forecasting_methods(
        series, horizon=4, methods=["seasonal_naive"],
    )
    spec = importlib.util.spec_from_file_location(
        "ts_agents.ui.components.forecasting",
        Path(__file__).parents[1] / "ts_agents/ui/components/forecasting.py",
    )
    forecasting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forecasting)
    monkeypatch.setattr(forecasting, "load_series_data", lambda *_: series)
    monkeypatch.setattr(comparison_module, "compare_forecasting_methods", lambda *_args, **_kwargs: comparison)
    monkeypatch.setattr(comparison_module, "plot_forecast_comparison", lambda *_: None)
    gradio.reset_mock()
    forecasting.create_forecasting_tab(gradio.State())
    compare = next(
        call.args[0] for call in gradio.Button.return_value.click.call_args_list
        if call.args[0].__name__ == "run_comparison"
    )
    _, metrics, _, _, status, _ = compare("run", "value", 4, second)
    assert status.startswith("Comparison complete")
    assert metrics.iloc[0]["RMSE"] == "0.0000"
    assert metrics.iloc[0]["MAPE (%)"] == "N/A (zero validation targets)"


def test_hosted_app_env_flag_helper(monkeypatch):
    sys.modules.pop("ts_agents.hosted_app", None)
    module = importlib.import_module("ts_agents.hosted_app")

    monkeypatch.setenv("TS_AGENTS_ENABLE_AGENT", "true")
    assert module._env_flag("TS_AGENTS_ENABLE_AGENT", False) is True

    monkeypatch.setenv("TS_AGENTS_ENABLE_AGENT", "0")
    assert module._env_flag("TS_AGENTS_ENABLE_AGENT", True) is False
