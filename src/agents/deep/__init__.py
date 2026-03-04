"""Deep Agent - Multi-agent architecture with specialized sub-agents.

This module provides a hierarchical agent system where an orchestrator
delegates to specialized sub-agents for complex analysis tasks:

- decomposition-agent: Trend/seasonal decomposition
- forecasting-agent: Model selection and prediction
- patterns-agent: Motif/anomaly detection
- classification-agent: TSC algorithm selection
- turbulence-agent: CFD/domain-specific analysis

Example usage:
    >>> from ts_agents.agents.deep import create_deep_agent, DeepAgentChat
    >>>
    >>> # Create the deep agent
    >>> agent = create_deep_agent()
    >>> result = agent.invoke({
    ...     "messages": [{"role": "user", "content": "Analyze the spectral properties"}]
    ... })
    >>>
    >>> # Or use the chat interface
    >>> chat = DeepAgentChat()
    >>> response = chat.chat("Decompose bx001_real and forecast the next 50 steps")
    >>> print(response)

Features:
- Automatic delegation to specialized sub-agents
- Cost-based approval workflow for expensive operations
- Filesystem backend for persistence
- Fallback to LangChain if deepagents not available
"""

from .orchestrator import (
    create_deep_agent,
    DeepAgentChat,
    DeepAgentTurn,
    SubagentCall,
    list_subagents,
    get_expensive_tools,
    run_with_approval,
    get_all_subagents,
    create_interrupt_config,
)

from .subagents import (
    DECOMPOSITION_SUBAGENT,
    FORECASTING_SUBAGENT,
    PATTERNS_SUBAGENT,
    CLASSIFICATION_SUBAGENT,
    TURBULENCE_SUBAGENT,
)

__all__ = [
    # Main API
    "create_deep_agent",
    "DeepAgentChat",

    # Data classes
    "DeepAgentTurn",
    "SubagentCall",

    # Utilities
    "list_subagents",
    "get_expensive_tools",
    "run_with_approval",
    "get_all_subagents",
    "create_interrupt_config",

    # Subagent definitions
    "DECOMPOSITION_SUBAGENT",
    "FORECASTING_SUBAGENT",
    "PATTERNS_SUBAGENT",
    "CLASSIFICATION_SUBAGENT",
    "TURBULENCE_SUBAGENT",
]
