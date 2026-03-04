"""Agent implementations for time series analysis.

This package provides:
- simple: LangChain-based agent with configurable tool bundles
- deep: Multi-agent architecture with specialized sub-agents
- benchmarks: Infrastructure for testing agent performance

Example usage:
    >>> # Simple agent
    >>> from ts_agents.agents.simple import create_simple_agent
    >>> agent = create_simple_agent(tool_bundle="standard")
    >>> result = agent.invoke({"messages": [{"role": "user", "content": "Analyze peaks"}]})
    >>>
    >>> # Deep agent with sub-agents
    >>> from ts_agents.agents.deep import create_deep_agent, DeepAgentChat
    >>> chat = DeepAgentChat()
    >>> response = chat.chat("Decompose bx001_real using the best method")
"""

from .simple import create_simple_agent, SimpleAgentChat
from .deep import create_deep_agent, DeepAgentChat, list_subagents
from .benchmarks import AgentBenchmark, BenchmarkResult

__all__ = [
    # Simple agent
    "create_simple_agent",
    "SimpleAgentChat",

    # Deep agent
    "create_deep_agent",
    "DeepAgentChat",
    "list_subagents",

    # Benchmarks
    "AgentBenchmark",
    "BenchmarkResult",
]
