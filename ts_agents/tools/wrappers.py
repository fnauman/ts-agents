"""Tool Wrappers - Convert core functions to agent-compatible tools.

This module provides functions to wrap core analysis functions
for use with different agent frameworks:
- LangChain tools
- deepagent tools
"""

from typing import Callable, List, Any, Dict, Optional, Sequence, Union
import functools
import inspect
import numpy as np
from pydantic import BaseModel, Field, create_model

from .registry import ToolRegistry, ToolMetadata, ToolCategory, ToolParameter


# =============================================================================
# LangChain Wrappers
# =============================================================================

def _create_args_schema(metadata: ToolMetadata) -> type[BaseModel]:
    """Create a Pydantic model for tool arguments from metadata.

    Parameters
    ----------
    metadata : ToolMetadata
        Tool metadata with parameter definitions

    Returns
    -------
    type[BaseModel]
        A Pydantic model class representing the tool's arguments
    """
    field_definitions = {}

    for param in metadata.parameters:
        # Map parameter types to Python types
        python_type = _param_type_to_python_type(param.type)

        if param.optional:
            if param.default is not None:
                field_definitions[param.name] = (
                    Optional[python_type],
                    Field(default=param.default, description=param.description)
                )
            else:
                field_definitions[param.name] = (
                    Optional[python_type],
                    Field(default=None, description=param.description)
                )
        else:
            field_definitions[param.name] = (
                python_type,
                Field(..., description=param.description)
            )

    # Create a dynamic Pydantic model
    model_name = f"{metadata.name.title().replace('_', '')}Args"
    return create_model(model_name, **field_definitions)


def _param_type_to_python_type(type_str: str) -> type:
    """Convert parameter type string to Python type."""
    mapping = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "np.ndarray": list,  # LangChain passes arrays as lists
        "numpy.ndarray": list,
        "ndarray": list,
        "array": list,
    }
    return mapping.get(type_str.lower(), Any)


def wrap_for_langchain(
    tool: Union[str, ToolMetadata],
    return_direct: bool = False,
) -> Any:
    """Wrap a registered tool for use with LangChain.

    Parameters
    ----------
    tool : str or ToolMetadata
        Tool name or metadata to wrap
    return_direct : bool
        Whether the tool should return directly to user

    Returns
    -------
    BaseTool
        LangChain-compatible tool

    Examples
    --------
    >>> from langchain.agents import create_openai_functions_agent
    >>> tool = wrap_for_langchain("stl_decompose")
    >>> agent = create_openai_functions_agent(llm, [tool], prompt)
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        raise ImportError(
            "LangChain not installed. Install with: pip install langchain-core"
        )

    if isinstance(tool, str):
        metadata = ToolRegistry.get(tool)
    else:
        metadata = tool

    def wrapped_func(**kwargs):
        """Execute the tool with preprocessed arguments."""
        processed_kwargs = _preprocess_args(kwargs, metadata)
        return metadata.core_function(**processed_kwargs)

    # Create args schema from metadata
    args_schema = _create_args_schema(metadata)

    # Create StructuredTool with proper schema
    return StructuredTool.from_function(
        func=wrapped_func,
        name=metadata.name,
        description=_format_description_for_llm(metadata),
        args_schema=args_schema,
        return_direct=return_direct,
    )


def wrap_tools_for_langchain(
    tools: Sequence[Union[str, ToolMetadata]],
    return_direct: bool = False,
) -> List[Any]:
    """Wrap multiple tools for LangChain.

    Parameters
    ----------
    tools : Sequence[str | ToolMetadata]
        Tool names or metadata objects
    return_direct : bool
        Whether tools should return directly to user

    Returns
    -------
    List[BaseTool]
        List of LangChain-compatible tools

    Examples
    --------
    >>> tools = wrap_tools_for_langchain(["stl_decompose", "detect_peaks"])
    """
    return [wrap_for_langchain(t, return_direct=return_direct) for t in tools]


def create_langchain_tool(
    func: Callable,
    name: str,
    description: str,
    return_direct: bool = False,
    args_schema: Optional[type[BaseModel]] = None,
) -> Any:
    """Create a LangChain tool from any function.

    Parameters
    ----------
    func : Callable
        Function to wrap
    name : str
        Tool name
    description : str
        Tool description
    return_direct : bool
        Whether to return directly
    args_schema : type[BaseModel], optional
        Pydantic model defining the tool's arguments. If None, inferred from
        the function signature.

    Returns
    -------
    BaseTool
        LangChain tool
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        raise ImportError(
            "LangChain not installed. Install with: pip install langchain-core"
        )

    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema,
        return_direct=return_direct,
    )


# =============================================================================
# deepagent Wrappers
# =============================================================================

def wrap_for_deepagent(
    tool: Union[str, ToolMetadata],
) -> Dict[str, Any]:
    """Wrap a registered tool for use with deepagents.

    Parameters
    ----------
    tool : str or ToolMetadata
        Tool name or metadata to wrap

    Returns
    -------
    dict
        deepagent-compatible tool definition

    Examples
    --------
    >>> from deepagents import create_deep_agent
    >>> tool = wrap_for_deepagent("stl_decompose")
    >>> agent = create_deep_agent(tools=[tool])
    """
    if isinstance(tool, str):
        metadata = ToolRegistry.get(tool)
    else:
        metadata = tool

    return {
        "name": metadata.name,
        "description": _format_description_for_llm(metadata),
        "function": _create_deepagent_function(metadata),
        "parameters": metadata.to_schema(),
    }


def wrap_tools_for_deepagent(
    tools: Sequence[Union[str, ToolMetadata]],
) -> List[Dict[str, Any]]:
    """Wrap multiple tools for deepagents.

    Parameters
    ----------
    tools : Sequence[str | ToolMetadata]
        Tool names or metadata objects

    Returns
    -------
    List[dict]
        List of deepagent-compatible tool definitions

    Examples
    --------
    >>> tools = wrap_tools_for_deepagent(["stl_decompose", "detect_peaks"])
    """
    return [wrap_for_deepagent(t) for t in tools]


def _create_deepagent_function(metadata: ToolMetadata) -> Callable:
    """Create a function wrapper for deepagent that handles result formatting."""

    @functools.wraps(metadata.core_function)
    def wrapper(**kwargs):
        # Convert any list inputs to numpy arrays if needed
        processed_kwargs = _preprocess_args(kwargs, metadata)

        # Call the core function
        result = metadata.core_function(**processed_kwargs)

        # Format the result for the LLM
        return _format_result_for_llm(result, metadata)

    return wrapper


# =============================================================================
# Generic Callable Wrappers
# =============================================================================

def create_callable_tool(
    tool: Union[str, ToolMetadata],
    format_result: bool = True,
) -> Callable:
    """Create a simple callable wrapper for a tool.

    This is useful when you don't need a specific framework's tool format,
    just a callable that formats inputs and outputs appropriately.

    Parameters
    ----------
    tool : str or ToolMetadata
        Tool name or metadata
    format_result : bool
        Whether to format the result as a string

    Returns
    -------
    Callable
        Wrapped function

    Examples
    --------
    >>> decompose = create_callable_tool("stl_decompose")
    >>> result = decompose(series=data, period=150)
    """
    if isinstance(tool, str):
        metadata = ToolRegistry.get(tool)
    else:
        metadata = tool

    @functools.wraps(metadata.core_function)
    def wrapper(**kwargs):
        processed_kwargs = _preprocess_args(kwargs, metadata)
        result = metadata.core_function(**processed_kwargs)
        if format_result:
            return _format_result_for_llm(result, metadata)
        return result

    return wrapper


# =============================================================================
# Helper Functions
# =============================================================================

def _format_description_for_llm(metadata: ToolMetadata) -> str:
    """Format tool description for LLM consumption.

    Includes description, parameter info, and examples.
    """
    parts = [metadata.description]

    # Add parameter documentation
    if metadata.parameters:
        parts.append("\nParameters:")
        for param in metadata.parameters:
            opt = "(optional)" if param.optional else "(required)"
            parts.append(f"  - {param.name} {opt}: {param.description}")

    # Add return info
    if metadata.returns:
        parts.append(f"\nReturns: {metadata.returns}")

    # Add examples if present
    if metadata.examples:
        parts.append("\nExample prompts:")
        for ex in metadata.examples[:2]:  # Limit to 2 examples
            parts.append(f"  - {ex}")

    return "\n".join(parts)


def _preprocess_args(kwargs: Dict[str, Any], metadata: ToolMetadata) -> Dict[str, Any]:
    """Preprocess arguments for a tool call.

    Handles common conversions like list -> numpy array.
    """
    result = {}

    for key, value in kwargs.items():
        # Convert lists to numpy arrays for array parameters
        if isinstance(value, list):
            # Check if this parameter expects an array
            for param in metadata.parameters:
                if param.name == key and param.type in ("np.ndarray", "ndarray", "array"):
                    value = np.array(value)
                    break

        result[key] = value

    return result


def _format_result_for_llm(result: Any, metadata: ToolMetadata) -> str:
    """Format analysis result for LLM consumption.

    Converts dataclass results to readable strings.
    """
    from .results import ToolResult, format_result

    if isinstance(result, ToolResult):
        return format_result(result)

    # Handle simple types
    if isinstance(result, (int, float)):
        return str(result)

    if isinstance(result, str):
        return result

    if isinstance(result, np.ndarray):
        if result.size <= 10:
            return f"Array: {result.tolist()}"
        return f"Array of shape {result.shape}, values from {result.min():.4f} to {result.max():.4f}"

    # Handle dataclass results (our AnalysisResult subclasses)
    if hasattr(result, 'to_dict'):
        return _format_dataclass_result(result)

    if hasattr(result, '__dict__'):
        return _format_dataclass_result(result)

    # Handle dict
    if isinstance(result, dict):
        return _format_dict_result(result)

    # Fallback
    return str(result)


def _format_dataclass_result(result: Any) -> str:
    """Format a dataclass result for LLM output."""
    parts = [f"## {result.__class__.__name__}"]

    for key, value in result.__dict__.items():
        if key.startswith('_'):
            continue

        # Handle numpy arrays specially
        if isinstance(value, np.ndarray):
            if value.size <= 5:
                formatted = str(value.tolist())
            else:
                formatted = f"array of {value.size} values (min={value.min():.4f}, max={value.max():.4f})"
        elif isinstance(value, float):
            formatted = f"{value:.6g}"
        elif isinstance(value, list) and len(value) > 5:
            formatted = f"list of {len(value)} items"
        else:
            formatted = str(value)

        parts.append(f"- {key}: {formatted}")

    return "\n".join(parts)


def _format_dict_result(result: Dict[str, Any]) -> str:
    """Format a dictionary result for LLM output."""
    parts = []
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            if value.size <= 5:
                formatted = str(value.tolist())
            else:
                formatted = f"array({value.size} values)"
        elif isinstance(value, float):
            formatted = f"{value:.6g}"
        else:
            formatted = str(value)
        parts.append(f"- {key}: {formatted}")

    return "\n".join(parts)


# =============================================================================
# Batch Tool Creation
# =============================================================================

def create_all_langchain_tools(
    categories: Optional[List[ToolCategory]] = None,
    max_cost: Optional[Any] = None,
) -> List[Any]:
    """Create LangChain tools for all registered tools matching criteria.

    Parameters
    ----------
    categories : List[ToolCategory], optional
        Filter by categories
    max_cost : ComputationalCost, optional
        Maximum computational cost

    Returns
    -------
    List[BaseTool]
        All matching LangChain tools
    """
    from .registry import ComputationalCost

    tools = []

    for metadata in ToolRegistry.list_all():
        # Category filter
        if categories is not None and metadata.category not in categories:
            continue

        # Cost filter
        if max_cost is not None:
            from .registry import _COST_ORDER
            if _COST_ORDER.index(metadata.cost) > _COST_ORDER.index(max_cost):
                continue

        tools.append(wrap_for_langchain(metadata))

    return tools


def create_all_deepagent_tools(
    categories: Optional[List[ToolCategory]] = None,
    max_cost: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Create deepagent tools for all registered tools matching criteria.

    Parameters
    ----------
    categories : List[ToolCategory], optional
        Filter by categories
    max_cost : ComputationalCost, optional
        Maximum computational cost

    Returns
    -------
    List[dict]
        All matching deepagent tools
    """
    from .registry import ComputationalCost

    tools = []

    for metadata in ToolRegistry.list_all():
        # Category filter
        if categories is not None and metadata.category not in categories:
            continue

        # Cost filter
        if max_cost is not None:
            from .registry import _COST_ORDER
            if _COST_ORDER.index(metadata.cost) > _COST_ORDER.index(max_cost):
                continue

        tools.append(wrap_for_deepagent(metadata))

    return tools
