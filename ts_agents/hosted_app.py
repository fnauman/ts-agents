"""Hosted Gradio entrypoint for installed-package deployments."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ts_agents.ui import create_app, launch_app

if TYPE_CHECKING:
    import gradio as gr

_app: "gr.Blocks | None" = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_app() -> "gr.Blocks":
    global _app
    if _app is None:
        _app = create_app(
            enable_agent=_env_flag("TS_AGENTS_ENABLE_AGENT", False),
            agent_type=os.environ.get("TS_AGENTS_AGENT_TYPE", "simple"),
            persist_sessions=_env_flag("TS_AGENTS_PERSIST_SESSIONS", False),
            title=os.environ.get("TS_AGENTS_UI_TITLE", "ts-agents Demo"),
        )
    return _app


def main() -> None:
    launch_app(
        get_app(),
        share=_env_flag("GRADIO_SHARE", False),
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
    )


def __getattr__(name: str):
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
