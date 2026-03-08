"""Tests for the tool registry."""

import inspect
import pytest
import numpy as np


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_registry_initialization(self):
        """Test that registry initializes with default tools."""
        from src.tools.registry import ToolRegistry

        # Force re-initialization
        ToolRegistry._initialized = False
        ToolRegistry._tools = {}

        # Access should trigger initialization
        tools = ToolRegistry.list_all()
        assert len(tools) > 0
        assert ToolRegistry._initialized is True

    def test_get_tool_by_name(self):
        """Test getting a tool by name."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("stl_decompose")
        assert tool.name == "stl_decompose"
        assert tool.description is not None
        assert callable(tool.core_function)

    def test_compare_forecasts_with_data_has_models_and_alias(self):
        """Test compare_forecasts_with_data parameters include models + seasonal controls."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("compare_forecasts_with_data")
        param_names = [param.name for param in tool.parameters]

        assert "models" in param_names
        assert "methods" in param_names
        assert "season_length" in param_names

    def test_forecast_seasonal_naive_with_data_has_season_length(self):
        """Seasonal naive should be registered with its seasonal period control."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("forecast_seasonal_naive_with_data")
        param_names = [param.name for param in tool.parameters]

        assert "season_length" in param_names

    def test_segment_changepoint_with_data_has_expected_params(self):
        """Test segment_changepoint_with_data exposes core controls + compatibility alias."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("segment_changepoint_with_data")
        param_names = [param.name for param in tool.parameters]

        assert "n_segments" in param_names
        assert "n_changepoints" in param_names
        assert "algorithm" in param_names
        assert "cost_model" in param_names
        assert "penalty" in param_names
        assert "min_size" in param_names

    def test_compute_coherence_with_data_has_sample_rate_aliases(self):
        """Test compute_coherence_with_data exposes sample_rate and compatibility aliases."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("compute_coherence_with_data")
        param_names = [param.name for param in tool.parameters]
        params_by_name = {param.name: param for param in tool.parameters}

        assert "sample_rate" in param_names
        assert "sampling_rate" in param_names
        assert "fs" in param_names
        assert params_by_name["sample_rate"].default is None

    def test_compute_psd_with_data_uses_consistent_wrapper_name(self):
        """PSD with-data tool should point at the correctly named wrapper."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("compute_psd_with_data")

        assert tool.core_function.__name__ == "compute_psd_with_data"

    def test_with_data_registry_params_match_wrapper_signatures(self):
        """All _with_data registry params should be accepted by wrapper signatures."""
        from src.tools.registry import ToolRegistry

        mismatches = []
        for tool in ToolRegistry.list_all():
            if not tool.name.endswith("_with_data"):
                continue

            sig = inspect.signature(tool.core_function)
            signature_params = sig.parameters
            has_varkw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature_params.values()
            )
            if has_varkw:
                continue

            missing = [
                p.name for p in tool.parameters
                if p.name not in signature_params
            ]
            if missing:
                mismatches.append((tool.name, missing))

        assert mismatches == []

    def test_get_nonexistent_tool(self):
        """Test that getting nonexistent tool raises KeyError."""
        from src.tools.registry import ToolRegistry

        with pytest.raises(KeyError):
            ToolRegistry.get("nonexistent_tool_xyz")

    def test_get_optional(self):
        """Test get_optional returns None for missing tools."""
        from src.tools.registry import ToolRegistry

        assert ToolRegistry.get_optional("nonexistent_tool") is None
        assert ToolRegistry.get_optional("stl_decompose") is not None

    def test_list_by_category(self):
        """Test listing tools by category."""
        from src.tools.registry import ToolRegistry, ToolCategory

        decomp_tools = ToolRegistry.list_by_category(ToolCategory.DECOMPOSITION)
        assert len(decomp_tools) >= 4  # STL, MSTL, HP, Holt-Winters

        for tool in decomp_tools:
            assert tool.category == ToolCategory.DECOMPOSITION

    def test_list_by_max_cost(self):
        """Test listing tools by maximum cost."""
        from src.tools.registry import ToolRegistry, ComputationalCost

        low_cost = ToolRegistry.list_by_max_cost(ComputationalCost.LOW)
        medium_cost = ToolRegistry.list_by_max_cost(ComputationalCost.MEDIUM)

        assert len(low_cost) > 0
        assert len(medium_cost) >= len(low_cost)

    def test_list_by_exact_cost(self):
        """Test listing tools by exact cost."""
        from src.tools.registry import ToolRegistry, ComputationalCost

        low_tools = ToolRegistry.list_by_cost(ComputationalCost.LOW)

        for tool in low_tools:
            assert tool.cost == ComputationalCost.LOW

    def test_search_tools(self):
        """Test searching tools by name/description."""
        from src.tools.registry import ToolRegistry

        results = ToolRegistry.search("decompose")
        assert len(results) > 0

        for tool in results:
            assert "decompose" in tool.name.lower() or "decompose" in tool.description.lower()

    def test_search_with_category_filter(self):
        """Test searching with category filter."""
        from src.tools.registry import ToolRegistry, ToolCategory

        results = ToolRegistry.search(
            "forecast",
            category=ToolCategory.FORECASTING,
        )

        for tool in results:
            assert tool.category == ToolCategory.FORECASTING

    def test_tool_has_required_metadata(self):
        """Test that all tools have required metadata."""
        from src.tools.registry import ToolRegistry

        for tool in ToolRegistry.list_all():
            assert tool.name, "Tool must have a name"
            assert tool.description, "Tool must have a description"
            assert tool.category is not None, "Tool must have a category"
            assert tool.cost is not None, "Tool must have a cost"
            assert callable(tool.core_function), "Tool must have a callable function"

    def test_tool_to_schema(self):
        """Test converting tool metadata to JSON schema."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("detect_peaks")
        schema = tool.to_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_category_summary(self):
        """Test getting category summary."""
        from src.tools.registry import ToolRegistry

        summary = ToolRegistry.get_tools_for_category_summary()

        assert "decomposition" in summary
        assert "forecasting" in summary
        assert "patterns" in summary

        for category, tools in summary.items():
            assert isinstance(tools, list)
            assert len(tools) > 0


class TestToolMetadata:
    """Tests for ToolMetadata class."""

    def test_get_signature(self):
        """Test getting function signature."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("stl_decompose")
        sig = tool.get_signature()

        assert "stl_decompose" in sig
        assert "series" in sig

    def test_to_schema_types(self):
        """Test JSON schema type conversion."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("stl_decompose")
        schema = tool.to_schema()

        # Check that types are JSON schema types
        for prop in schema["properties"].values():
            assert prop["type"] in ["string", "number", "integer", "boolean", "array", "object"]


class TestComputationalCost:
    """Tests for computational cost ordering."""

    def test_cost_ordering(self):
        """Test that costs are properly ordered."""
        from src.tools.registry import ComputationalCost, _COST_ORDER

        assert _COST_ORDER.index(ComputationalCost.LOW) < _COST_ORDER.index(ComputationalCost.MEDIUM)
        assert _COST_ORDER.index(ComputationalCost.MEDIUM) < _COST_ORDER.index(ComputationalCost.HIGH)
        assert _COST_ORDER.index(ComputationalCost.HIGH) < _COST_ORDER.index(ComputationalCost.VERY_HIGH)


class TestToolExecution:
    """Tests for actually executing registered tools."""

    def test_execute_detect_peaks(self):
        """Test executing the detect_peaks tool."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("detect_peaks")

        # Create test data with clear peaks
        x = np.sin(np.linspace(0, 10 * np.pi, 1000))

        result = tool.core_function(x, distance=50)

        assert result.count > 0
        assert len(result.peak_indices) == result.count

    def test_execute_stl_decompose(self):
        """Test executing the stl_decompose tool."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("stl_decompose")

        # Create test data
        t = np.linspace(0, 10, 1000)
        x = 0.5 * t + np.sin(2 * np.pi * t) + 0.1 * np.random.randn(1000)

        result = tool.core_function(x, period=100)

        assert result.method == "stl"
        assert len(result.trend) == 1000

    def test_execute_describe_series(self):
        """Test executing the describe_series tool."""
        from src.tools.registry import ToolRegistry

        tool = ToolRegistry.get("describe_series")

        x = np.random.randn(1000)
        result = tool.core_function(x)

        assert result.length == 1000
        assert result.mean is not None
        assert result.std is not None
