"""Hosted Gradio entrypoint for source-checkout deployments."""

from ts_agents.hosted_app import get_app, main

app = get_app()


if __name__ == "__main__":
    main()
