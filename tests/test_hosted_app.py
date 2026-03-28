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


def test_hosted_app_env_flag_helper(monkeypatch):
    sys.modules.pop("ts_agents.hosted_app", None)
    module = importlib.import_module("ts_agents.hosted_app")

    monkeypatch.setenv("TS_AGENTS_ENABLE_AGENT", "true")
    assert module._env_flag("TS_AGENTS_ENABLE_AGENT", False) is True

    monkeypatch.setenv("TS_AGENTS_ENABLE_AGENT", "0")
    assert module._env_flag("TS_AGENTS_ENABLE_AGENT", True) is False
