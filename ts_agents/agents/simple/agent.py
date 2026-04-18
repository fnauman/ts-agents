"""Simple LangChain-based agent with configurable tool bundles.

This agent serves as a testing platform for understanding how LLMs
perform with different numbers and combinations of tools.

Example usage:
    >>> from ts_agents.agents.simple import create_simple_agent
    >>>
    >>> # Create agent with different tool bundles
    >>> agent_minimal = create_simple_agent(tool_bundle="minimal")  # 5 tools
    >>> agent_standard = create_simple_agent(tool_bundle="standard")  # 15 tools
    >>> agent_full = create_simple_agent(tool_bundle="full")  # 25+ tools
    >>>
    >>> # Run a query
    >>> result = agent_standard.invoke({
    ...     "messages": [{"role": "user", "content": "How many peaks in bx001_real?"}]
    ... })
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from ...config import get_openai_model
from ...tools.bundles import (
    get_bundle,
    get_bundle_names,
    get_langchain_bundle,
    list_available_bundles,
)
from ...tools.registry import ToolRegistry
from .prompts import get_system_prompt, get_bundle_prompt


# Configure logging
logger = logging.getLogger(__name__)


def _get_chat_openai():
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Simple agent support requires optional dependencies. Install with: pip install "ts-agents[agents]"'
        ) from exc
    return ChatOpenAI


def _get_message_types():
    try:
        from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Simple agent support requires optional dependencies. Install with: pip install "ts-agents[agents]"'
        ) from exc
    return HumanMessage, ToolMessage, AIMessage


# =============================================================================
# Agent Creation
# =============================================================================

def create_simple_agent(
    model_name: Optional[str] = None,
    tool_bundle: str = "standard",
    custom_tools: Optional[List[str]] = None,
    temperature: float = 0,
    include_data_info: bool = False,
    enable_logging: bool = True,
    capture_results: bool = False,
    log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    *,
    data_context_prompt: Optional[str] = None,
) -> Any:
    """Create a simple LangChain agent for time series analysis.

    Parameters
    ----------
    model_name : str, optional
        OpenAI model name. Defaults to config setting.
    tool_bundle : str
        Predefined tool bundle:
        - "minimal": 5 core tools (baseline testing)
        - "standard": 15 tools (typical analysis)
        - "full": 25+ tools (comprehensive)
        - "all": Every registered tool
        Or a category name like "decomposition", "forecasting", etc.
    custom_tools : List[str], optional
        Override with specific tool names (ignores tool_bundle)
    temperature : float
        LLM temperature (0 = deterministic)
    include_data_info : bool
        Include the bundled CFD/MHD data context in the system prompt
    enable_logging : bool
        Enable logging of tool calls and decisions
    capture_results : bool
        Capture JSON-serializable tool results in log entries (may be large)
    log_callback : Callable, optional
        Custom callback for logging events
    data_context_prompt : str, optional
        Explicit domain or dataset context to append to the system prompt

    Returns
    -------
    Agent
        LangChain agent ready for invocation

    Examples
    --------
    >>> # Test with minimal tools (5)
    >>> agent = create_simple_agent(tool_bundle="minimal")
    >>>
    >>> # Test with full toolkit (25+)
    >>> agent = create_simple_agent(tool_bundle="full")
    >>>
    >>> # Custom tool selection
    >>> agent = create_simple_agent(custom_tools=["stl_decompose", "detect_peaks"])
    """
    try:
        from langchain.agents import create_agent
    except ModuleNotFoundError as exc:
        raise ImportError(
            'Simple agent support requires optional dependencies. Install with: pip install "ts-agents[agents]"'
        ) from exc

    model_name = model_name or get_openai_model()

    # Get tools
    if custom_tools:
        tool_names = custom_tools
        langchain_tools = []
        for name in custom_tools:
            metadata = ToolRegistry.get(name)
            from ...tools.wrappers import wrap_for_langchain
            langchain_tools.append(wrap_for_langchain(metadata))
        bundle_name = "custom"
    else:
        tool_names = get_bundle_names(tool_bundle)
        langchain_tools = get_langchain_bundle(tool_bundle)
        bundle_name = tool_bundle

    # Validate that we have at least one tool
    if not langchain_tools:
        raise ValueError(
            f"Tool bundle '{bundle_name}' returned no tools. "
            f"Available bundles: {list_available_bundles()}"
        )

    # Generate system prompt
    system_prompt = get_system_prompt(
        tool_names=tool_names,
        include_data_info=include_data_info,
        data_context_prompt=data_context_prompt,
    )
    bundle_prompt = get_bundle_prompt(bundle_name)
    full_prompt = system_prompt + "\n" + bundle_prompt

    # Create LLM
    llm = _get_chat_openai()(model=model_name, temperature=temperature)

    # Wrap tools with logging if enabled
    if enable_logging:
        langchain_tools = _wrap_tools_with_logging(
            langchain_tools,
            log_callback=log_callback,
            capture_results=capture_results,
        )

    # Create agent using the new function-calling API
    agent = create_agent(
        model=llm,
        tools=langchain_tools,
        system_prompt=full_prompt,
    )

    # Store metadata on the agent for benchmarking
    agent._ts_agents_metadata = {
        "model_name": model_name,
        "tool_bundle": bundle_name,
        "tool_count": len(langchain_tools),
        "tool_names": tool_names,
        "created_at": datetime.now().isoformat(),
    }

    if enable_logging:
        logger.info(
            f"Created agent with {len(langchain_tools)} tools "
            f"(bundle={bundle_name}, model={model_name})"
        )

    return agent


def _wrap_tools_with_logging(
    tools: List[Any],
    log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    capture_results: bool = False,
) -> List[Any]:
    """Wrap tools to log their invocations.

    This enables analysis of tool selection patterns.
    """
    wrapped_tools = []

    for tool in tools:
        original_func = tool.func

        def create_logged_func(orig_func, tool_name):
            def logged_func(*args, **kwargs):
                start_time = time.time()
                log_entry = {
                    "event": "tool_call",
                    "tool_name": tool_name,
                    "timestamp": datetime.now().isoformat(),
                    "args": str(args)[:200],
                    "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                }

                try:
                    result = orig_func(*args, **kwargs)
                    log_entry["status"] = "success"
                    log_entry["duration_ms"] = (time.time() - start_time) * 1000
                    log_entry["result_preview"] = str(result)[:200]
                    if capture_results:
                        from ts_agents.cli.output import to_jsonable
                        log_entry["result_payload"] = to_jsonable(result)
                except Exception as e:
                    log_entry["status"] = "error"
                    log_entry["error"] = str(e)
                    log_entry["duration_ms"] = (time.time() - start_time) * 1000
                    logger.error(f"Tool {tool_name} failed: {e}")
                    raise

                logger.debug(f"Tool call: {log_entry}")

                if log_callback:
                    log_callback(log_entry)

                return result

            return logged_func

        tool.func = create_logged_func(original_func, tool.name)
        wrapped_tools.append(tool)

    return wrapped_tools


# =============================================================================
# Chat Interface
# =============================================================================

@dataclass
class ToolCallRecord:
    """Record of a single tool call."""
    tool_name: str
    args: Dict[str, Any]
    result: Optional[str] = None
    result_payload: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationTurn:
    """Record of a single conversation turn."""
    user_message: str
    assistant_response: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    duration_ms: float = 0.0
    token_count: Optional[int] = None


class SimpleAgentChat:
    """Chat interface for the simple agent with conversation history and logging.

    This class provides:
    - Conversation history management
    - Logging of all tool calls
    - Statistics on tool usage
    - Session export for analysis

    Parameters
    ----------
    model_name : str, optional
        OpenAI model name
    tool_bundle : str
        Tool bundle to use
    custom_tools : List[str], optional
        Custom tool list
    enable_logging : bool
        Enable detailed logging
    capture_results : bool
        Capture JSON-serializable tool outputs in the session log
    data_context_prompt : str, optional
        Explicit domain or dataset context to pass through to create_simple_agent

    Examples
    --------
    >>> chat = SimpleAgentChat(tool_bundle="standard")
    >>> response = chat.chat("How many peaks in bx001_real for Re200Rm200?")
    >>> print(response)
    >>>
    >>> # Get tool usage statistics
    >>> stats = chat.get_tool_stats()
    >>> print(f"Tools used: {stats['tool_call_count']}")
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        tool_bundle: str = "standard",
        custom_tools: Optional[List[str]] = None,
        enable_logging: bool = True,
        capture_results: bool = False,
        *,
        data_context_prompt: Optional[str] = None,
    ):
        self._tool_calls: List[ToolCallRecord] = []

        def log_callback(entry: Dict[str, Any]):
            if entry.get("event") == "tool_call":
                self._tool_calls.append(ToolCallRecord(
                    tool_name=entry["tool_name"],
                    args=entry.get("kwargs", {}),
                    result=entry.get("result_preview"),
                    result_payload=entry.get("result_payload"),
                    error=entry.get("error"),
                    duration_ms=entry.get("duration_ms", 0),
                    timestamp=entry.get("timestamp", ""),
                ))

        self.agent = create_simple_agent(
            model_name=model_name,
            tool_bundle=tool_bundle,
            custom_tools=custom_tools,
            enable_logging=enable_logging,
            capture_results=capture_results,
            log_callback=log_callback,
            data_context_prompt=data_context_prompt,
        )

        self.messages: List[Any] = []
        self.turns: List[ConversationTurn] = []
        self.tool_bundle = tool_bundle
        self._session_start = datetime.now()

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
        start_time = time.time()
        tool_calls_before = len(self._tool_calls)
        HumanMessage, _, _ = _get_message_types()

        self.messages.append(HumanMessage(content=user_message))

        result = self.agent.invoke({"messages": self.messages})

        # Update message history
        self.messages = result.get("messages", self.messages)

        # Extract response
        response = self._extract_response()

        # Record turn
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=response,
            tool_calls=self._tool_calls[tool_calls_before:],
            duration_ms=(time.time() - start_time) * 1000,
        )
        self.turns.append(turn)

        return response

    def _extract_response(self) -> str:
        """Extract the final response from messages."""
        image_data = ""
        _, ToolMessage, AIMessage = _get_message_types()

        for msg in self.messages:
            if isinstance(msg, ToolMessage) and hasattr(msg, 'content'):
                if '[IMAGE_DATA:' in msg.content:
                    import re
                    match = re.search(r'\[IMAGE_DATA:[A-Za-z0-9+/=]+\]', msg.content)
                    if match:
                        image_data = match.group(0)

        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage) and msg.content:
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    response = msg.content
                    if image_data and '[IMAGE_DATA:' not in response:
                        return f"{response} {image_data}"
                    return response

        return "No response generated."

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.turns = []
        self._tool_calls = []
        self._session_start = datetime.now()

    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics on tool usage in this session.

        Returns
        -------
        Dict[str, Any]
            Statistics including:
            - tool_call_count: Total number of tool calls
            - tools_used: Set of unique tools used
            - tool_frequency: Count per tool
            - avg_tool_duration_ms: Average tool call duration
            - error_count: Number of failed tool calls
        """
        tool_frequency: Dict[str, int] = {}
        total_duration = 0.0
        error_count = 0

        for call in self._tool_calls:
            tool_frequency[call.tool_name] = tool_frequency.get(call.tool_name, 0) + 1
            total_duration += call.duration_ms
            if call.error:
                error_count += 1

        return {
            "tool_call_count": len(self._tool_calls),
            "tools_used": set(tool_frequency.keys()),
            "tool_frequency": tool_frequency,
            "avg_tool_duration_ms": total_duration / len(self._tool_calls) if self._tool_calls else 0,
            "error_count": error_count,
            "conversation_turns": len(self.turns),
        }

    def get_session_data(self) -> Dict[str, Any]:
        """Get complete session data for analysis/export.

        Returns
        -------
        Dict[str, Any]
            Complete session data including:
            - metadata: Agent configuration
            - turns: All conversation turns
            - tool_calls: All tool calls
            - stats: Usage statistics
        """
        return {
            "metadata": {
                "tool_bundle": self.tool_bundle,
                "agent_metadata": getattr(self.agent, '_ts_agents_metadata', {}),
                "session_start": self._session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
            },
            "turns": [
                {
                    "user_message": turn.user_message,
                    "assistant_response": turn.assistant_response,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "args": tc.args,
                            "result": tc.result,
                            "result_payload": tc.result_payload,
                            "error": tc.error,
                            "duration_ms": tc.duration_ms,
                        }
                        for tc in turn.tool_calls
                    ],
                    "duration_ms": turn.duration_ms,
                }
                for turn in self.turns
            ],
            "stats": self.get_tool_stats(),
        }

    def export_session(self, filepath: str) -> None:
        """Export session data to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the JSON file
        """
        import json
        from pathlib import Path

        data = self.get_session_data()

        # Convert sets to lists for JSON serialization
        if "stats" in data and "tools_used" in data["stats"]:
            data["stats"]["tools_used"] = list(data["stats"]["tools_used"])

        Path(filepath).write_text(json.dumps(data, indent=2))
        logger.info(f"Session exported to {filepath}")


# =============================================================================
# Utility Functions
# =============================================================================

def run_single_query(
    query: str,
    tool_bundle: str = "standard",
    model_name: Optional[str] = None,
) -> str:
    """Convenience function to run a single query.

    Parameters
    ----------
    query : str
        The user's query
    tool_bundle : str
        Tool bundle to use
    model_name : str, optional
        Model name override

    Returns
    -------
    str
        The agent's response
    """
    agent = create_simple_agent(
        model_name=model_name,
        tool_bundle=tool_bundle,
        enable_logging=False,
    )
    HumanMessage, _, AIMessage = _get_message_types()

    result = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "No response generated."


def compare_bundles_on_query(
    query: str,
    bundles: Optional[List[str]] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare how different tool bundles handle the same query.

    Parameters
    ----------
    query : str
        Query to test
    bundles : List[str], optional
        Bundles to compare (default: minimal, standard, full)
    model_name : str, optional
        Model name override

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Results for each bundle including response and stats
    """
    if bundles is None:
        bundles = ["minimal", "standard", "full"]

    results = {}

    for bundle in bundles:
        chat = SimpleAgentChat(
            model_name=model_name,
            tool_bundle=bundle,
        )

        start_time = time.time()
        response = chat.chat(query)
        elapsed = time.time() - start_time

        results[bundle] = {
            "response": response,
            "elapsed_seconds": elapsed,
            "stats": chat.get_tool_stats(),
            "tool_count": len(get_bundle_names(bundle)),
        }

    return results
