import importlib
import sys

import gradio as gr


def test_hosted_app_exports_gradio_blocks(monkeypatch):
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

    sys.modules.pop("app", None)
    module = importlib.import_module("app")

    assert isinstance(module.app, gr.Blocks)


def test_hosted_app_env_flag_helper(monkeypatch):
    sys.modules.pop("app", None)
    module = importlib.import_module("app")

    monkeypatch.setenv("TS_AGENTS_ENABLE_AGENT", "true")
    assert module._env_flag("TS_AGENTS_ENABLE_AGENT", False) is True

    monkeypatch.setenv("TS_AGENTS_ENABLE_AGENT", "0")
    assert module._env_flag("TS_AGENTS_ENABLE_AGENT", True) is False
