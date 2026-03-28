"""Deep Agent - Multi-agent architecture with specialized sub-agents."""

from __future__ import annotations

from ts_agents._lazy import load_export

_LAZY_EXPORTS = {
    "create_deep_agent": ("orchestrator", "create_deep_agent"),
    "DeepAgentChat": ("orchestrator", "DeepAgentChat"),
    "DeepAgentTurn": ("orchestrator", "DeepAgentTurn"),
    "SubagentCall": ("orchestrator", "SubagentCall"),
    "list_subagents": ("orchestrator", "list_subagents"),
    "get_expensive_tools": ("orchestrator", "get_expensive_tools"),
    "run_with_approval": ("orchestrator", "run_with_approval"),
    "get_all_subagents": ("orchestrator", "get_all_subagents"),
    "create_interrupt_config": ("orchestrator", "create_interrupt_config"),
    "DECOMPOSITION_SUBAGENT": ("subagents", "DECOMPOSITION_SUBAGENT"),
    "FORECASTING_SUBAGENT": ("subagents", "FORECASTING_SUBAGENT"),
    "PATTERNS_SUBAGENT": ("subagents", "PATTERNS_SUBAGENT"),
    "CLASSIFICATION_SUBAGENT": ("subagents", "CLASSIFICATION_SUBAGENT"),
    "TURBULENCE_SUBAGENT": ("subagents", "TURBULENCE_SUBAGENT"),
}


def __getattr__(name: str):
    value = load_export(__name__, _LAZY_EXPORTS, name)
    globals()[name] = value
    return value

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
