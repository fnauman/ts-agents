"""Tools Layer - Tool registration, wrapping, and bundling for agents.

This module provides:
- ToolRegistry: Central registry for all analysis tools
- Tool wrappers for LangChain and deepagents
- Predefined tool bundles for different use cases

Quick Start
-----------
>>> from ts_agents.tools import ToolRegistry, get_bundle
>>>
>>> # List all available tools
>>> tools = ToolRegistry.list_all()
>>> print(f"{len(tools)} tools registered")
>>>
>>> # Get a predefined bundle
>>> standard_tools = get_bundle("standard")
>>>
>>> # Get tools for LangChain
>>> from ts_agents.tools import get_langchain_bundle
>>> langchain_tools = get_langchain_bundle("standard")

Examples
--------
>>> # Get tool metadata
>>> from ts_agents.tools import ToolRegistry
>>> tool = ToolRegistry.get("stl_decompose")
>>> print(tool.description)
>>> print(f"Cost: {tool.cost.value}")

>>> # Create LangChain agent with tools
>>> from ts_agents.tools import get_langchain_bundle
>>> tools = get_langchain_bundle("minimal")
>>> # agent = create_openai_functions_agent(llm, tools, prompt)

>>> # Create deepagent with tools
>>> from ts_agents.tools import get_deepagent_bundle
>>> tools = get_deepagent_bundle("standard")
>>> # agent = create_deep_agent(tools=tools)
"""

from .registry import (
    ToolRegistry,
    ToolMetadata,
    ToolParameter,
    ToolCategory,
    ComputationalCost,
)

from .bundles import (
    get_bundle,
    get_bundle_names,
    list_available_bundles,
    get_bundle_summary,
    create_custom_bundle,
    get_subagent_bundle,
    get_langchain_bundle,
    get_deepagent_bundle,
    # Bundle name constants
    MINIMAL_BUNDLE,
    STANDARD_BUNDLE,
    FULL_BUNDLE,
    ORCHESTRATOR_BUNDLE,
    CATEGORY_BUNDLES,
)

from .wrappers import (
    wrap_for_langchain,
    wrap_tools_for_langchain,
    wrap_for_deepagent,
    wrap_tools_for_deepagent,
    create_callable_tool,
    create_all_langchain_tools,
    create_all_deepagent_tools,
)

from .executor import (
    ToolExecutor,
    ToolError,
    ToolErrorCode,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    SandboxMode,
    get_executor,
    execute_tool,
)

from .results import (
    ToolResult,
    Visualization,
    DecompositionResult,
    ForecastResult,
    PeakResult,
    SpectralResult,
    MotifResult,
    DiscordResult,
    MatrixProfileResult,
    ChangePointResult,
    StatisticsResult,
    ScalarResult,
    GenericResult,
    ResultFormatter,
    get_formatter,
    format_result,
)

__all__ = [
    # Registry
    "ToolRegistry",
    "ToolMetadata",
    "ToolParameter",
    "ToolCategory",
    "ComputationalCost",
    # Bundles
    "get_bundle",
    "get_bundle_names",
    "list_available_bundles",
    "get_bundle_summary",
    "create_custom_bundle",
    "get_subagent_bundle",
    "get_langchain_bundle",
    "get_deepagent_bundle",
    # Bundle constants
    "MINIMAL_BUNDLE",
    "STANDARD_BUNDLE",
    "FULL_BUNDLE",
    "ORCHESTRATOR_BUNDLE",
    "CATEGORY_BUNDLES",
    # Wrappers
    "wrap_for_langchain",
    "wrap_tools_for_langchain",
    "wrap_for_deepagent",
    "wrap_tools_for_deepagent",
    "create_callable_tool",
    "create_all_langchain_tools",
    "create_all_deepagent_tools",
    # Executor
    "ToolExecutor",
    "ToolError",
    "ToolErrorCode",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionStatus",
    "SandboxMode",
    "get_executor",
    "execute_tool",
    # Results
    "ToolResult",
    "Visualization",
    "DecompositionResult",
    "ForecastResult",
    "PeakResult",
    "SpectralResult",
    "MotifResult",
    "DiscordResult",
    "MatrixProfileResult",
    "ChangePointResult",
    "StatisticsResult",
    "ScalarResult",
    "GenericResult",
    "ResultFormatter",
    "get_formatter",
    "format_result",
]
