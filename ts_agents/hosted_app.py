"""Hosted Gradio entrypoint for installed-package deployments."""

from __future__ import annotations

import os

from ts_agents.ui import create_app, launch_app


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


app = create_app(
    enable_agent=_env_flag("TS_AGENTS_ENABLE_AGENT", False),
    agent_type=os.environ.get("TS_AGENTS_AGENT_TYPE", "simple"),
    persist_sessions=_env_flag("TS_AGENTS_PERSIST_SESSIONS", False),
    title=os.environ.get("TS_AGENTS_UI_TITLE", "ts-agents Demo"),
)


def main() -> None:
    launch_app(
        app,
        share=_env_flag("GRADIO_SHARE", False),
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
