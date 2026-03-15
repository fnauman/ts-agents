"""Deep Agent Orchestrator - Multi-agent architecture with specialized sub-agents.

This module implements a hierarchical agent system where an orchestrator
delegates specialized tasks to domain-specific sub-agents:
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
    >>>
    >>> # Or use the chat interface
    >>> chat = DeepAgentChat()
    >>> response = chat.chat("Analyze the spectral properties of bx001_real")
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ...config import get_openai_model, get_results_cache_dir
from ...tools.registry import ToolRegistry, ComputationalCost, _COST_ORDER
from ...tools.bundles import get_bundle, get_subagent_bundle
from ...tools.wrappers import wrap_tools_for_deepagent

from .subagents import (
    DECOMPOSITION_SUBAGENT,
    FORECASTING_SUBAGENT,
    PATTERNS_SUBAGENT,
    CLASSIFICATION_SUBAGENT,
    TURBULENCE_SUBAGENT,
)
from .subagents.decomposition import get_decomposition_tools
from .subagents.forecasting import get_forecasting_tools
from .subagents.patterns import get_patterns_tools
from .subagents.classification import get_classification_tools
from .subagents.turbulence import get_turbulence_tools


logger = logging.getLogger(__name__)


# =============================================================================
# System Prompts
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are a time series analysis orchestrator.

Your role is to understand user requests and delegate to specialized sub-agents
when appropriate. You have access to high-level tools for quick analysis and
can delegate complex tasks to specialists.

## Your Sub-Agents

You can delegate to these specialists using the task tool:

1. **decomposition-agent**: For trend/seasonal decomposition
   - Use when: Separating trend, seasonality, residuals
   - Methods: STL, MSTL, Holt-Winters

2. **forecasting-agent**: For time series prediction
   - Use when: Predicting future values
   - Methods: ARIMA, ETS, Theta, Ensemble

3. **patterns-agent**: For pattern and anomaly detection
   - Use when: Finding motifs, discords, peaks, segments
   - Methods: Matrix profile, peak detection, RQA, FLUSS

4. **classification-agent**: For time series classification
   - Use when: Categorizing time series
   - Methods: DTW-KNN, ROCKET, HIVE-COTE 2

5. **turbulence-agent**: For CFD/turbulence-specific analysis
   - Use when: Analyzing MHD/dynamo simulations
   - Methods: PSD, coherence, spectral slopes

## Your Approach

1. **Understand the request**: What does the user want to achieve?

2. **Quick analysis**: Use your tools for simple requests:
   - Basic statistics: describe_series_with_data
   - Quick decomposition: stl_decompose_with_data
   - Ensemble forecast: forecast_ensemble_with_data
   - Pattern overview: analyze_matrix_profile_with_data

3. **Complex tasks**: Delegate to specialists when:
   - User needs method comparison/selection
   - Domain expertise is required
   - Multi-step analysis needed

4. **Synthesize results**: After delegation:
   - Summarize key findings
   - Provide actionable insights
   - Suggest follow-up analysis

## Available Data

CFD/MHD simulation data at different Reynolds numbers:
- Runs: Re200Rm200, Re175Rm175, Re150Rm150, Re125Rm125, Re105Rm105, Re102_5Rm102_5
- Variables: bx001_real, by001_real, vx001_imag, vy001_imag, ex001_imag, ey001_imag

## Important Notes

- **Cost awareness**: Some tools are expensive (marked VERY_HIGH cost)
- **Results persistence**: Check /results/ for cached analyses
- **Confidence intervals**: Always report uncertainty when forecasting
- **Visualization**: Include plots when helpful

## Response Format

Structure your responses clearly:
1. What you understood from the request
2. Analysis approach taken
3. Key findings and results
4. Recommendations or next steps
"""


# =============================================================================
# Cost-Based Approval
# =============================================================================

def get_expensive_tool_names() -> List[str]:
    """Get names of tools that require approval due to high cost."""
    expensive = ToolRegistry.list_by_cost(ComputationalCost.VERY_HIGH)
    return [t.name for t in expensive]


def create_interrupt_config(enable_approval: bool = True) -> Optional[Dict[str, bool]]:
    """Create interrupt configuration for expensive tools.

    Parameters
    ----------
    enable_approval : bool
        Whether to require approval for expensive operations

    Returns
    -------
    dict or None
        Mapping of tool names to interrupt flags
    """
    if not enable_approval:
        return None

    expensive_tools = get_expensive_tool_names()
    if not expensive_tools:
        return None

    return {tool_name: True for tool_name in expensive_tools}


# =============================================================================
# Sub-Agent Configuration
# =============================================================================

def _build_subagent_config(
    subagent_dict: Dict[str, Any],
    tools_func: Callable[[], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build a complete subagent configuration with tools.

    Parameters
    ----------
    subagent_dict : dict
        Base subagent configuration (name, description, system_prompt)
    tools_func : callable
        Function that returns wrapped tools for this subagent

    Returns
    -------
    dict
        Complete subagent configuration with tools
    """
    config = subagent_dict.copy()
    config["tools"] = tools_func()
    return config


def get_all_subagents() -> List[Dict[str, Any]]:
    """Get all subagent configurations with their tools.

    Returns
    -------
    List[dict]
        List of complete subagent configurations
    """
    return [
        _build_subagent_config(DECOMPOSITION_SUBAGENT, get_decomposition_tools),
        _build_subagent_config(FORECASTING_SUBAGENT, get_forecasting_tools),
        _build_subagent_config(PATTERNS_SUBAGENT, get_patterns_tools),
        _build_subagent_config(CLASSIFICATION_SUBAGENT, get_classification_tools),
        _build_subagent_config(TURBULENCE_SUBAGENT, get_turbulence_tools),
    ]


# =============================================================================
# Agent Creation
# =============================================================================

def create_deep_agent(
    model_name: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    enable_approval: bool = True,
    enable_logging: bool = True,
    custom_system_prompt: Optional[str] = None,
) -> Any:
    """Create a deep agent with specialized sub-agents.

    This creates a hierarchical agent system where the orchestrator can
    delegate to specialized sub-agents for complex analysis tasks.

    Parameters
    ----------
    model_name : str, optional
        Model to use. Defaults to config setting.
    workspace_dir : str, optional
        Directory for agent workspace. Defaults to results cache dir.
    enable_approval : bool
        Whether to require approval for expensive operations (VERY_HIGH cost)
    enable_logging : bool
        Enable logging of agent actions
    custom_system_prompt : str, optional
        Override the default orchestrator system prompt

    Returns
    -------
    Agent
        Deep agent ready for invocation

    Examples
    --------
    >>> agent = create_deep_agent()
    >>> result = agent.invoke({
    ...     "messages": [{"role": "user", "content": "Analyze spectral slopes"}]
    ... })

    >>> # Without approval workflow (for testing)
    >>> agent = create_deep_agent(enable_approval=False)
    """
    model_name = model_name or get_openai_model()
    workspace_dir = workspace_dir or str(get_results_cache_dir())

    # Get orchestrator tools
    orchestrator_bundle = get_bundle("orchestrator")
    orchestrator_tools = wrap_tools_for_deepagent(orchestrator_bundle)

    # Get all subagent configurations
    subagents = get_all_subagents()

    # Create interrupt configuration for expensive tools
    interrupt_on = create_interrupt_config(enable_approval)

    # System prompt
    system_prompt = custom_system_prompt or ORCHESTRATOR_SYSTEM_PROMPT

    # Try to create with deepagents if available
    try:
        return _create_with_deepagents(
            model_name=model_name,
            tools=orchestrator_tools,
            subagents=subagents,
            system_prompt=system_prompt,
            interrupt_on=interrupt_on,
            workspace_dir=workspace_dir,
            enable_logging=enable_logging,
        )
    except ImportError:
        logger.warning(
            "deepagents not available, falling back to LangChain-based implementation"
        )
        return _create_with_langchain(
            model_name=model_name,
            tools=orchestrator_tools,
            subagents=subagents,
            system_prompt=system_prompt,
            enable_approval=enable_approval,
            enable_logging=enable_logging,
        )


def _create_with_deepagents(
    model_name: str,
    tools: List[Dict[str, Any]],
    subagents: List[Dict[str, Any]],
    system_prompt: str,
    interrupt_on: Optional[Dict[str, bool]],
    workspace_dir: str,
    enable_logging: bool,
) -> Any:
    """Create agent using deepagents library."""
    from deepagents import create_deep_agent as deepagent_create
    from deepagents.middleware import FilesystemMiddleware

    # Configure filesystem middleware for persistence
    middleware = [FilesystemMiddleware(workspace_dir)]

    agent = deepagent_create(
        model=model_name,
        tools=tools,
        subagents=subagents,
        system_prompt=system_prompt,
        middleware=middleware,
        interrupt_on=interrupt_on,
        debug=enable_logging,
    )

    # Store metadata
    agent._ts_agents_metadata = {
        "agent_type": "deep",
        "model_name": model_name,
        "subagent_count": len(subagents),
        "subagent_names": [s["name"] for s in subagents],
        "enable_approval": interrupt_on is not None,
        "workspace_dir": workspace_dir,
        "created_at": datetime.now().isoformat(),
    }

    if enable_logging:
        logger.info(
            f"Created deep agent with {len(subagents)} sub-agents "
            f"(model={model_name}, approval={'enabled' if interrupt_on else 'disabled'})"
        )

    return agent


def _create_with_langchain(
    model_name: str,
    tools: List[Dict[str, Any]],
    subagents: List[Dict[str, Any]],
    system_prompt: str,
    enable_approval: bool,
    enable_logging: bool,
) -> Any:
    """Create agent using LangChain as fallback.

    This is a simplified version that doesn't have full subagent delegation
    but provides similar functionality through tool expansion.
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool as langchain_tool

    llm = ChatOpenAI(model=model_name, temperature=0)

    # Convert deepagent tools to LangChain format
    # Note: We use a factory function to avoid closure capturing issues
    def make_tool_wrapper(td):
        """Create a wrapper function that captures the tool_dict by value."""
        @langchain_tool
        def wrapped(**kwargs):
            return td["function"](**kwargs)
        wrapped.name = td["name"]
        wrapped.description = td["description"]
        return wrapped

    langchain_tools = []
    for tool_dict in tools:
        langchain_tools.append(make_tool_wrapper(tool_dict))

    # Add subagent tools (flattened, since we can't delegate)
    def make_subagent_tool_wrapper(td, subagent_name):
        """Create a wrapper for subagent tools."""
        @langchain_tool
        def wrapped(**kwargs):
            return td["function"](**kwargs)
        wrapped.name = td["name"]
        wrapped.description = f"[{subagent_name}] {td['description']}"
        return wrapped

    for subagent in subagents:
        subagent_tools = subagent.get("tools", [])
        for tool_dict in subagent_tools:
            langchain_tools.append(make_subagent_tool_wrapper(tool_dict, subagent["name"]))

    # Create prompt
    enhanced_prompt = system_prompt + """

Note: Running in fallback mode. Sub-agent delegation is simulated through
direct tool access. All sub-agent tools are available directly.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_prompt),
        ("placeholder", "{messages}"),
    ])

    agent = create_agent(
        model=llm,
        tools=langchain_tools,
        system_prompt=prompt,
    )

    # Store metadata
    agent._ts_agents_metadata = {
        "agent_type": "deep_fallback",
        "model_name": model_name,
        "subagent_count": len(subagents),
        "subagent_names": [s["name"] for s in subagents],
        "total_tools": len(langchain_tools),
        "enable_approval": enable_approval,
        "created_at": datetime.now().isoformat(),
    }

    if enable_logging:
        logger.info(
            f"Created deep agent (fallback mode) with {len(langchain_tools)} tools "
            f"(model={model_name})"
        )

    return agent


# =============================================================================
# Chat Interface
# =============================================================================

@dataclass
class SubagentCall:
    """Record of a subagent delegation."""
    subagent_name: str
    query: str
    response: str
    duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DeepAgentTurn:
    """Record of a conversation turn with the deep agent."""
    user_message: str
    assistant_response: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    subagent_calls: List[SubagentCall] = field(default_factory=list)
    duration_ms: float = 0.0
    required_approval: bool = False


class DeepAgentChat:
    """Chat interface for the deep agent with conversation history.

    This class provides:
    - Conversation history management
    - Tracking of tool and subagent usage
    - Session export for analysis

    Parameters
    ----------
    model_name : str, optional
        Model to use
    enable_approval : bool
        Whether to require approval for expensive operations
    enable_logging : bool
        Enable detailed logging

    Examples
    --------
    >>> chat = DeepAgentChat()
    >>> response = chat.chat("Decompose bx001_real using STL")
    >>> print(response)
    >>>
    >>> # Get usage stats
    >>> stats = chat.get_stats()
    >>> print(f"Subagent delegations: {stats['subagent_calls']}")
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        enable_approval: bool = True,
        enable_logging: bool = True,
    ):
        self.agent = create_deep_agent(
            model_name=model_name,
            enable_approval=enable_approval,
            enable_logging=enable_logging,
        )

        self.messages: List[Any] = []
        self.turns: List[DeepAgentTurn] = []
        self._session_start = datetime.now()
        self._tool_calls: List[Dict[str, Any]] = []
        self._subagent_calls: List[SubagentCall] = []

    def chat(self, user_message: str) -> str:
        """Send a message and get a response.

        Parameters
        ----------
        user_message : str
            The user's message

        Returns
        -------
        str
            The agent's response
        """
        from langchain_core.messages import HumanMessage, AIMessage

        start_time = time.time()
        tool_calls_before = len(self._tool_calls)
        subagent_calls_before = len(self._subagent_calls)

        self.messages.append(HumanMessage(content=user_message))

        result = self.agent.invoke({"messages": self.messages})

        # Update message history
        self.messages = result.get("messages", self.messages)

        # Extract response
        response = self._extract_response()

        # Record turn
        turn = DeepAgentTurn(
            user_message=user_message,
            assistant_response=response,
            tool_calls=self._tool_calls[tool_calls_before:],
            subagent_calls=self._subagent_calls[subagent_calls_before:],
            duration_ms=(time.time() - start_time) * 1000,
        )
        self.turns.append(turn)

        return response

    def _extract_response(self) -> str:
        """Extract the final response from messages."""
        from langchain_core.messages import AIMessage

        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage) and msg.content:
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    return msg.content

        return "No response generated."

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.turns = []
        self._tool_calls = []
        self._subagent_calls = []
        self._session_start = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics on agent usage.

        Returns
        -------
        Dict[str, Any]
            Usage statistics
        """
        subagent_frequency: Dict[str, int] = {}
        for call in self._subagent_calls:
            subagent_frequency[call.subagent_name] = (
                subagent_frequency.get(call.subagent_name, 0) + 1
            )

        return {
            "conversation_turns": len(self.turns),
            "tool_calls": len(self._tool_calls),
            "subagent_calls": len(self._subagent_calls),
            "subagent_frequency": subagent_frequency,
            "total_duration_ms": sum(t.duration_ms for t in self.turns),
        }

    def get_session_data(self) -> Dict[str, Any]:
        """Get complete session data for export.

        Returns
        -------
        Dict[str, Any]
            Complete session data
        """
        return {
            "metadata": {
                "agent_type": "deep",
                "agent_metadata": getattr(self.agent, '_ts_agents_metadata', {}),
                "session_start": self._session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
            },
            "turns": [
                {
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "tool_calls": turn.tool_calls,
                    "subagent_calls": [
                        {
                            "subagent_name": sc.subagent_name,
                            "query": sc.query,
                            "response": sc.response[:500] if sc.response else None,
                            "duration_ms": sc.duration_ms,
                        }
                        for sc in turn.subagent_calls
                    ],
                    "duration_ms": turn.duration_ms,
                }
                for turn in self.turns
            ],
            "stats": self.get_stats(),
        }

    def export_session(self, filepath: str) -> None:
        """Export session data to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the JSON file
        """
        import json

        data = self.get_session_data()
        Path(filepath).write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Session exported to {filepath}")


# =============================================================================
# Utility Functions
# =============================================================================

def list_subagents() -> List[Dict[str, str]]:
    """List available sub-agents and their capabilities.

    Returns
    -------
    List[Dict[str, str]]
        List of subagent info with name and description
    """
    subagents = get_all_subagents()
    return [
        {"name": s["name"], "description": s["description"]}
        for s in subagents
    ]


def get_expensive_tools() -> List[Dict[str, Any]]:
    """Get information about tools that require approval.

    Returns
    -------
    List[Dict[str, Any]]
        List of expensive tool information
    """
    expensive = ToolRegistry.list_by_cost(ComputationalCost.VERY_HIGH)
    return [
        {
            "name": t.name,
            "description": t.description,
            "category": t.category.value,
        }
        for t in expensive
    ]


def run_with_approval(
    agent: Any,
    query: str,
    approval_callback: Optional[Callable[[str], bool]] = None,
) -> str:
    """Run a query with approval workflow for expensive operations.

    Parameters
    ----------
    agent : Agent
        The deep agent
    query : str
        User query
    approval_callback : callable, optional
        Function that returns True to approve expensive operation.
        If None, auto-approves all.

    Returns
    -------
    str
        Agent response
    """
    from langchain_core.messages import HumanMessage

    result = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    # Check if approval is needed (if using deepagents with interrupt)
    if hasattr(result, 'interrupt') and result.interrupt:
        tool_name = result.interrupt.get('tool_name', 'unknown')
        if approval_callback is None or approval_callback(tool_name):
            # Continue with approval
            result = agent.invoke(None, resume=True)
        else:
            return f"Operation '{tool_name}' was not approved."

    # Extract response
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            return msg.content

    return "No response generated."
