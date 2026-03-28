"""Lightweight CLI launcher for the optional Gradio UI."""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import Sequence


_UI_INSTALL_HINT = 'Gradio UI requires optional dependencies. Install with: pip install "ts-agents[ui]"'


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for the UI launcher."""
    parser = argparse.ArgumentParser(description="Time Series Analysis UI")
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Disable agent chat tab",
    )
    parser.add_argument(
        "--agent-type",
        choices=["simple", "deep"],
        default="simple",
        help="Default agent type (default: simple)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable session persistence",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    return parser


def _load_ui_runtime():
    """Import the UI runtime only after argument parsing succeeds."""
    try:
        return import_module("ts_agents.ui.gradio_app")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "gradio":
            raise ImportError(_UI_INSTALL_HINT) from exc
        raise


def main(argv: Sequence[str] | None = None) -> None:
    """Run the optional Gradio UI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        ui_runtime = _load_ui_runtime()
    except ImportError as exc:
        parser.exit(1, f"{exc}\n")
        return

    app = ui_runtime.create_app(
        enable_agent=not args.no_agent,
        agent_type=args.agent_type,
        persist_sessions=not args.no_persist,
    )
    ui_runtime.launch_app(
        app,
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
