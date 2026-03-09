"""Tests for tool wrappers."""

import pytest
import numpy as np


class TestHelperFunctions:
    """Tests for wrapper helper functions."""

    def test_format_description(self):
        """Test formatting tool description for LLM."""
        from ts_agents.tools.registry import ToolRegistry
        from ts_agents.tools.wrappers import _format_description_for_llm

        tool = ToolRegistry.get("stl_decompose")
        description = _format_description_for_llm(tool)

        assert tool.name in description or tool.description in description
        assert "Parameters:" in description
        assert "Returns:" in description

    def test_preprocess_args_list_to_array(self):
        """Test that list arguments are converted to numpy arrays."""
        from ts_agents.tools.registry import ToolRegistry
        from ts_agents.tools.wrappers import _preprocess_args

        tool = ToolRegistry.get("detect_peaks")

        # Input as list
        kwargs = {"series": [1, 2, 3, 4, 5]}
        processed = _preprocess_args(kwargs, tool)

        assert isinstance(processed["series"], np.ndarray)
        assert list(processed["series"]) == [1, 2, 3, 4, 5]

    def test_args_schema_does_not_force_sample_rate_for_coherence_with_data(self):
        """Legacy aliases should not be shadowed by an injected sample_rate default."""
        from ts_agents.tools.registry import ToolRegistry
        from ts_agents.tools.wrappers import _create_args_schema

        tool = ToolRegistry.get("compute_coherence_with_data")
        args_schema = _create_args_schema(tool)

        if hasattr(args_schema, "model_fields"):
            sample_rate_default = args_schema.model_fields["sample_rate"].default
            data = args_schema(
                variable1="a",
                unique_id1="run1",
                variable2="b",
                unique_id2="run2",
                fs=2.5,
            ).model_dump()
        else:  # pragma: no cover - pydantic v1 compatibility
            sample_rate_default = args_schema.__fields__["sample_rate"].default
            data = args_schema(
                variable1="a",
                unique_id1="run1",
                variable2="b",
                unique_id2="run2",
                fs=2.5,
            ).dict()

        assert sample_rate_default is None
        assert data["sample_rate"] is None
        assert data["fs"] == 2.5

    def test_format_result_simple_types(self):
        """Test formatting simple result types."""
        from ts_agents.tools.registry import ToolRegistry
        from ts_agents.tools.wrappers import _format_result_for_llm

        tool = ToolRegistry.get("count_peaks")

        # Integer result
        assert _format_result_for_llm(42, tool) == "42"

        # Float result
        assert "3.14" in _format_result_for_llm(3.14159, tool)

        # String result
        assert _format_result_for_llm("test", tool) == "test"

    def test_format_result_array(self):
        """Test formatting numpy array results."""
        from ts_agents.tools.registry import ToolRegistry
        from ts_agents.tools.wrappers import _format_result_for_llm

        tool = ToolRegistry.get("detect_peaks")

        # Small array
        small = np.array([1, 2, 3])
        result = _format_result_for_llm(small, tool)
        assert "[1, 2, 3]" in result

        # Large array
        large = np.random.randn(100)
        result = _format_result_for_llm(large, tool)
        assert "shape" in result or str(large.min()) in result

    def test_format_dataclass_result(self):
        """Test formatting dataclass results."""
        from ts_agents.tools.registry import ToolRegistry
        from ts_agents.tools.wrappers import _format_result_for_llm

        tool = ToolRegistry.get("detect_peaks")

        # Create a simple test series with peaks
        x = np.sin(np.linspace(0, 10 * np.pi, 500))
        result = tool.core_function(x, distance=30)

        formatted = _format_result_for_llm(result, tool)

        assert "PeakResult" in formatted
        assert "count" in formatted


class TestCallableWrapper:
    """Tests for creating callable wrappers."""

    def test_create_callable_tool(self):
        """Test creating a callable tool wrapper."""
        from ts_agents.tools.wrappers import create_callable_tool

        detect_peaks = create_callable_tool("detect_peaks", format_result=False)

        x = np.sin(np.linspace(0, 10 * np.pi, 500))
        result = detect_peaks(series=x, distance=30)

        assert result.count > 0
        assert len(result.peak_indices) == result.count

    def test_create_callable_tool_with_formatting(self):
        """Test callable tool with result formatting."""
        from ts_agents.tools.wrappers import create_callable_tool

        detect_peaks = create_callable_tool("detect_peaks", format_result=True)

        x = np.sin(np.linspace(0, 10 * np.pi, 500))
        result = detect_peaks(series=x, distance=30)

        # Result should be a formatted string
        assert isinstance(result, str)
        assert "PeakResult" in result


class TestDeepagentWrapper:
    """Tests for deepagent wrappers."""

    def test_wrap_for_deepagent(self):
        """Test wrapping tool for deepagent."""
        from ts_agents.tools.wrappers import wrap_for_deepagent

        tool_def = wrap_for_deepagent("detect_peaks")

        assert "name" in tool_def
        assert "description" in tool_def
        assert "function" in tool_def
        assert "parameters" in tool_def

        assert tool_def["name"] == "detect_peaks"
        assert callable(tool_def["function"])

    def test_wrap_tools_for_deepagent(self):
        """Test wrapping multiple tools for deepagent."""
        from ts_agents.tools.wrappers import wrap_tools_for_deepagent

        tool_defs = wrap_tools_for_deepagent(["detect_peaks", "stl_decompose"])

        assert len(tool_defs) == 2
        names = [t["name"] for t in tool_defs]
        assert "detect_peaks" in names
        assert "stl_decompose" in names

    def test_deepagent_tool_execution(self):
        """Test executing a deepagent-wrapped tool."""
        from ts_agents.tools.wrappers import wrap_for_deepagent

        tool_def = wrap_for_deepagent("describe_series")

        x = np.random.randn(100)
        result = tool_def["function"](series=x)

        # Result should be formatted string
        assert isinstance(result, str)
        assert "mean" in result.lower() or "std" in result.lower()


class TestBatchCreation:
    """Tests for batch tool creation."""

    def test_create_all_deepagent_tools(self):
        """Test creating all deepagent tools."""
        from ts_agents.tools.wrappers import create_all_deepagent_tools
        from ts_agents.tools.registry import ToolRegistry

        tools = create_all_deepagent_tools()

        # Should have tools for all registered
        assert len(tools) == len(ToolRegistry.list_all())

    def test_create_deepagent_tools_with_category_filter(self):
        """Test creating deepagent tools with category filter."""
        from ts_agents.tools.wrappers import create_all_deepagent_tools
        from ts_agents.tools.registry import ToolCategory

        tools = create_all_deepagent_tools(categories=[ToolCategory.DECOMPOSITION])

        assert len(tools) >= 4  # At least 4 decomposition tools
        # All should be for decomposition category
        for tool in tools:
            assert "decompose" in tool["name"] or "filter" in tool["name"]

    def test_create_deepagent_tools_with_cost_filter(self):
        """Test creating deepagent tools with cost filter."""
        from ts_agents.tools.wrappers import create_all_deepagent_tools
        from ts_agents.tools.registry import ComputationalCost

        low_cost_tools = create_all_deepagent_tools(
            max_cost=ComputationalCost.LOW
        )

        # Should be fewer than all tools
        all_tools = create_all_deepagent_tools()
        assert len(low_cost_tools) < len(all_tools)


# Tests that require LangChain
try:
    from langchain_core.tools import BaseTool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
class TestLangChainWrapper:
    """Tests for LangChain wrappers (requires langchain)."""

    def test_wrap_for_langchain(self):
        """Test wrapping tool for LangChain."""
        from ts_agents.tools.wrappers import wrap_for_langchain

        tool = wrap_for_langchain("detect_peaks")

        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert tool.name == "detect_peaks"

    def test_wrap_tools_for_langchain(self):
        """Test wrapping multiple tools for LangChain."""
        from ts_agents.tools.wrappers import wrap_tools_for_langchain

        tools = wrap_tools_for_langchain(["detect_peaks", "stl_decompose"])

        assert len(tools) == 2

    def test_create_all_langchain_tools(self):
        """Test creating all LangChain tools."""
        from ts_agents.tools.wrappers import create_all_langchain_tools
        from ts_agents.tools.registry import ToolRegistry

        tools = create_all_langchain_tools()

        assert len(tools) == len(ToolRegistry.list_all())
