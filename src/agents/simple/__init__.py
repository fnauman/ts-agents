"""Simple LangChain-based agent for testing tool scaling.

This module provides a configurable agent that can be tested with
different tool bundles (minimal, standard, full, all) to understand
how agents perform with varying numbers of tools.

Example usage:
    >>> from ts_agents.agents.simple import create_simple_agent
    >>>
    >>> # Create agent with standard tool bundle (15 tools)
    >>> agent = create_simple_agent(tool_bundle="standard")
    >>>
    >>> # Create agent with minimal tools for faster testing
    >>> agent = create_simple_agent(tool_bundle="minimal")
    >>>
    >>> # Chat interface with history
    >>> chat = SimpleAgentChat(tool_bundle="full")
    >>> response = chat.chat("How many peaks in bx001_real?")
"""

from .agent import create_simple_agent, SimpleAgentChat

__all__ = [
    "create_simple_agent",
    "SimpleAgentChat",
]
