"""Main entry point for the Time Series Analysis Application.

This module provides the Gradio interface for time series analysis with:
- Agent chat (Simple or Deep agent)
- Manual analysis tabs (Decomposition, Forecasting, Patterns, Classification)
- Cross-run comparison
- Session persistence

Usage:
    # Run with default settings
    python main.py

    # Run with deep agent
    python main.py --agent-type deep

    # Run manual mode only (no agent)
    python main.py --no-agent

    # Create public share link
    python main.py --share

    # Custom port
    python main.py --port 8080
"""

from src.ui import create_app, launch_app


def main():
    """Launch the Gradio interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Time Series Analysis UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Default settings
  python main.py --agent-type deep  # Use deep agent
  python main.py --no-agent         # Manual mode only
  python main.py --share            # Create public link
  python main.py --port 8080        # Custom port
        """
    )
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

    print("Creating Time Series Analysis application...")
    app = create_app(
        enable_agent=not args.no_agent,
        agent_type=args.agent_type,
        persist_sessions=not args.no_persist,
    )

    print(f"Launching on http://{args.host}:{args.port}")
    launch_app(
        app,
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
