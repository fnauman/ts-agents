"""Main Gradio application for Time Series Analysis.

This module provides the main application factory and launch functions.
Supports both agent-based and manual analysis modes.

Example usage:
    >>> from ts_agents.ui import create_app, launch_app
    >>>
    >>> # Create and launch with default settings
    >>> app = create_app()
    >>> launch_app(app)
    >>>
    >>> # Create with specific configuration
    >>> app = create_app(enable_agent=True, agent_type="deep", persist_sessions=True)
    >>> launch_app(app, share=True)
"""

import gradio as gr
from typing import Optional

from .state import UIState, create_initial_state
from .components import (
    create_chat_tab,
    create_decomposition_tab,
    create_forecasting_tab,
    create_patterns_tab,
    create_classification_tab,
    create_comparison_tab,
)
from ..persistence import SessionStore, get_default_store


def create_app(
    enable_agent: bool = True,
    agent_type: str = "simple",
    persist_sessions: bool = True,
    title: str = "Time Series Analysis",
) -> gr.Blocks:
    """Create the Gradio application.

    Parameters
    ----------
    enable_agent : bool
        Enable the agent chat tab
    agent_type : str
        Default agent type: "simple" or "deep"
    persist_sessions : bool
        Enable session persistence
    title : str
        Application title

    Returns
    -------
    gr.Blocks
        Configured Gradio application

    Examples
    --------
    >>> app = create_app()
    >>> app.launch()

    >>> # With deep agent
    >>> app = create_app(agent_type="deep")
    >>> app.launch()

    >>> # Manual mode only (no agent)
    >>> app = create_app(enable_agent=False)
    >>> app.launch()
    """
    # Initialize persistence if enabled
    session_store = None
    if persist_sessions:
        try:
            from ..config import init_persistence
            init_persistence()
            session_store = get_default_store()
        except Exception as e:
            print(f"Warning: Failed to initialize persistence: {e}")

    # Gradio 6.0: theme/css moved to launch()
    with gr.Blocks(title=title) as app:
        # Application header
        gr.Markdown(f"""
        # {title}

        A comprehensive toolkit for time series analysis with both agent-based
        and manual analysis modes. Supports decomposition, forecasting, pattern
        detection, classification, and cross-run comparison.
        """)

        # Session state
        state = gr.State(create_initial_state())

        # Main tabs
        with gr.Tabs() as main_tabs:
            # Agent Chat Tab
            if enable_agent:
                with gr.Tab("Agent Chat", id="chat"):
                    create_chat_tab(state, agent_type=agent_type)

            # Manual Analysis Tabs
            with gr.Tab("Decomposition", id="decomposition"):
                create_decomposition_tab(state)

            with gr.Tab("Forecasting", id="forecasting"):
                create_forecasting_tab(state)

            with gr.Tab("Patterns", id="patterns"):
                create_patterns_tab(state)

            with gr.Tab("Classification", id="classification"):
                create_classification_tab(state)

            with gr.Tab("Compare", id="compare"):
                create_comparison_tab(state)

            # Settings/History Tab
            with gr.Tab("Settings & History", id="settings"):
                _create_settings_tab(state, session_store)

        # Session persistence callbacks
        if session_store:
            # Load latest session on app load
            def load_session():
                try:
                    return create_initial_state()
                except Exception:
                    return UIState()

            app.load(load_session, outputs=[state])

            # Save session on state change
            def save_session(ui_state: UIState):
                if ui_state is not None and session_store is not None:
                    try:
                        ui_state.save(session_store)
                    except Exception as e:
                        print(f"Warning: Failed to save session: {e}")
                return ui_state

            state.change(save_session, inputs=[state], outputs=[state])

    return app


def _create_settings_tab(state: gr.State, session_store: Optional[SessionStore]):
    """Create the settings and history tab."""
    gr.Markdown("""
    ## Settings & History

    View analysis history and manage sessions.
    """)

    with gr.Row():
        # Left column: Session management
        with gr.Column(scale=1):
            gr.Markdown("### Session Management")

            session_info = gr.Markdown("*No session loaded*")

            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                clear_btn = gr.Button("Clear History")
                new_session_btn = gr.Button("New Session")

            if session_store:
                gr.Markdown("### Recent Sessions")
                sessions_list = gr.Dataframe(
                    headers=["Session ID", "Updated", "Analyses"],
                    row_count=5,
                    interactive=False
                )

        # Right column: Analysis history
        with gr.Column(scale=2):
            gr.Markdown("### Analysis History")

            history_output = gr.Dataframe(
                headers=["Time", "Method", "Run", "Variable", "Summary"],
                row_count=10,
                interactive=False
            )

            gr.Markdown("### Export")
            with gr.Row():
                export_history_btn = gr.Button("Export History (JSON)")
                export_csv_btn = gr.Button("Export History (CSV)")

    def refresh_session_info(ui_state: UIState):
        """Refresh session information display."""
        if ui_state is None:
            if session_store:
                return "*No session loaded*", [], []
            return "*No session loaded*", []

        session = ui_state.session

        info = f"""
**Session ID**: {session.session_id}
**Created**: {session.created[:19]}
**Updated**: {session.updated[:19]}
**Current Run**: {session.current_run_id or 'None'}
**Current Variable**: {session.current_variable or 'None'}
**Analyses**: {len(session.analysis_history)}
**Chat Messages**: {len(session.chat_messages)}
"""

        # History table
        history_data = []
        for analysis in session.analysis_history[-20:]:  # Last 20
            history_data.append([
                analysis.get("timestamp", "")[:19],
                analysis.get("method", ""),
                analysis.get("run_id", ""),
                analysis.get("variable", ""),
                (analysis.get("result_summary", "") or "")[:50],
            ])

        # Sessions list
        sessions_data = []
        if session_store:
            for sess in session_store.list_sessions()[:10]:
                sessions_data.append([
                    sess.get("session_id", ""),
                    sess.get("updated", "")[:19],
                    str(sess.get("analysis_count", 0)),
                ])

        if session_store:
            return info, history_data, sessions_data
        return info, history_data

    def clear_history(ui_state: UIState):
        """Clear analysis history."""
        if ui_state is not None:
            ui_state.session.analysis_history = []
            ui_state.session.chat_messages = []
        return ui_state

    def new_session(ui_state: UIState):
        """Create a new session."""
        return UIState()

    def export_history_json(ui_state: UIState):
        """Export history as JSON."""
        if ui_state is None:
            return gr.Info("No session to export")

        try:
            import json
            from datetime import datetime

            data = {
                "session_id": ui_state.session.session_id,
                "created": ui_state.session.created,
                "updated": ui_state.session.updated,
                "analysis_history": ui_state.session.analysis_history,
                "chat_messages": ui_state.session.chat_messages,
            }

            filename = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

            return gr.Info(f"History exported to {filename}")
        except Exception as e:
            return gr.Warning(f"Export failed: {e}")

    def export_history_csv(ui_state: UIState):
        """Export history as CSV."""
        if ui_state is None:
            return gr.Info("No session to export")

        try:
            import pandas as pd
            from datetime import datetime

            if not ui_state.session.analysis_history:
                return gr.Info("No analysis history to export")

            df = pd.DataFrame(ui_state.session.analysis_history)

            filename = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)

            return gr.Info(f"History exported to {filename}")
        except Exception as e:
            return gr.Warning(f"Export failed: {e}")

    # Connect events
    refresh_btn.click(
        refresh_session_info,
        inputs=[state],
        outputs=[session_info, history_output, sessions_list] if session_store else [session_info, history_output]
    )

    clear_btn.click(
        clear_history,
        inputs=[state],
        outputs=[state]
    ).then(
        refresh_session_info,
        inputs=[state],
        outputs=[session_info, history_output, sessions_list] if session_store else [session_info, history_output]
    )

    new_session_btn.click(
        new_session,
        inputs=[state],
        outputs=[state]
    ).then(
        refresh_session_info,
        inputs=[state],
        outputs=[session_info, history_output, sessions_list] if session_store else [session_info, history_output]
    )

    export_history_btn.click(
        export_history_json,
        inputs=[state],
        outputs=[]
    )

    export_csv_btn.click(
        export_history_csv,
        inputs=[state],
        outputs=[]
    )

    # Initial refresh on tab selection
    # Note: This would require additional Gradio tab change event handling


def launch_app(
    app: gr.Blocks,
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    **kwargs
) -> None:
    """Launch the Gradio application.

    This function launches the app with Gradio 6.0 compatible settings.

    Parameters
    ----------
    app : gr.Blocks
        The Gradio application to launch
    share : bool
        Create a public share link
    server_name : str
        Server host address
    server_port : int
        Server port
    **kwargs
        Additional arguments passed to app.launch()

    Examples
    --------
    >>> app = create_app()
    >>> launch_app(app)

    >>> # With public share link
    >>> launch_app(app, share=True)

    >>> # Custom port
    >>> launch_app(app, server_port=8080)
    """
    # Gradio 6.0 compatible launch settings
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        # Gradio 6.0: footer links configuration
        # footer_links=["api"] if kwargs.pop("show_api", True) else [],
        **kwargs
    )


def main():
    """Main entry point for the Gradio application."""
    import argparse

    parser = argparse.ArgumentParser(description="Time Series Analysis UI")
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Disable agent chat tab"
    )
    parser.add_argument(
        "--agent-type",
        choices=["simple", "deep"],
        default="simple",
        help="Default agent type (default: simple)"
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable session persistence"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    app = create_app(
        enable_agent=not args.no_agent,
        agent_type=args.agent_type,
        persist_sessions=not args.no_persist,
    )

    launch_app(
        app,
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
