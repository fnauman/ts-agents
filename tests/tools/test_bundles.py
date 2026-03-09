"""Tests for tool bundles."""

import pytest


class TestBundles:
    """Tests for tool bundles."""

    def test_get_demo_meta_bundle(self):
        """Test that demo bundle combines windowing and forecasting demo tools."""
        from ts_agents.tools.bundles import (
            get_bundle_names,
            DEMO_BUNDLE,
            DEMO_WINDOWING_BUNDLE,
            DEMO_FORECASTING_BUNDLE,
        )

        names = get_bundle_names("demo")

        expected = list(dict.fromkeys(DEMO_WINDOWING_BUNDLE + DEMO_FORECASTING_BUNDLE))
        assert names == expected
        assert names == list(DEMO_BUNDLE)

    def test_get_demo_focused_bundles(self):
        """Test that focused demo bundles are available independently."""
        from ts_agents.tools.bundles import (
            get_bundle_names,
            DEMO_WINDOWING_BUNDLE,
            DEMO_FORECASTING_BUNDLE,
        )

        assert get_bundle_names("demo_windowing") == list(DEMO_WINDOWING_BUNDLE)
        assert get_bundle_names("demo_forecasting") == list(DEMO_FORECASTING_BUNDLE)

    def test_get_minimal_bundle(self):
        """Test getting minimal bundle."""
        from ts_agents.tools.bundles import get_bundle, MINIMAL_BUNDLE

        tools = get_bundle("minimal")

        assert len(tools) == len(MINIMAL_BUNDLE)
        tool_names = [t.name for t in tools]

        for name in MINIMAL_BUNDLE:
            assert name in tool_names

    def test_get_standard_bundle(self):
        """Test getting standard bundle."""
        from ts_agents.tools.bundles import get_bundle, STANDARD_BUNDLE

        tools = get_bundle("standard")

        assert len(tools) == len(STANDARD_BUNDLE)

    def test_get_full_bundle(self):
        """Test getting full bundle."""
        from ts_agents.tools.bundles import get_bundle, FULL_BUNDLE

        tools = get_bundle("full")

        assert len(tools) == len(FULL_BUNDLE)
        # Full should be bigger than standard
        assert len(FULL_BUNDLE) > len(get_bundle("standard"))

    def test_get_all_bundle(self):
        """Test getting all registered tools."""
        from ts_agents.tools.bundles import get_bundle
        from ts_agents.tools.registry import ToolRegistry

        all_tools = get_bundle("all")

        assert len(all_tools) == len(ToolRegistry.list_all())

    def test_get_orchestrator_bundle(self):
        """Test getting orchestrator bundle."""
        from ts_agents.tools.bundles import get_bundle, ORCHESTRATOR_BUNDLE

        tools = get_bundle("orchestrator")

        assert len(tools) == len(ORCHESTRATOR_BUNDLE)

    def test_get_category_bundle(self):
        """Test getting category-specific bundle."""
        from ts_agents.tools.bundles import get_bundle, CATEGORY_BUNDLES
        from ts_agents.tools.registry import ToolCategory

        for category_name in CATEGORY_BUNDLES.keys():
            tools = get_bundle(category_name)
            assert len(tools) > 0

    def test_get_bundle_invalid_name(self):
        """Test that invalid bundle name raises error."""
        from ts_agents.tools.bundles import get_bundle

        with pytest.raises(ValueError):
            get_bundle("nonexistent_bundle")

    def test_get_bundle_names(self):
        """Test getting tool names for a bundle."""
        from ts_agents.tools.bundles import get_bundle_names, MINIMAL_BUNDLE

        names = get_bundle_names("minimal")

        assert names == list(MINIMAL_BUNDLE)

    def test_list_available_bundles(self):
        """Test listing all available bundles."""
        from ts_agents.tools.bundles import list_available_bundles

        bundles = list_available_bundles()

        assert "demo" in bundles
        assert "demo_windowing" in bundles
        assert "demo_forecasting" in bundles
        assert "minimal" in bundles
        assert "standard" in bundles
        assert "full" in bundles
        assert "all" in bundles
        assert "orchestrator" in bundles

    def test_get_bundle_summary(self):
        """Test getting bundle summary."""
        from ts_agents.tools.bundles import get_bundle_summary

        summary = get_bundle_summary()

        assert "demo" in summary
        assert "demo_windowing" in summary
        assert "demo_forecasting" in summary
        assert "minimal" in summary
        assert "standard" in summary
        assert "full" in summary

        for name, info in summary.items():
            assert "count" in info
            assert "description" in info
            assert "tools" in info
            assert info["count"] == len(info["tools"])


class TestCustomBundles:
    """Tests for custom bundle creation."""

    def test_create_custom_bundle_by_tools(self):
        """Test creating custom bundle with specific tools."""
        from ts_agents.tools.bundles import create_custom_bundle

        bundle = create_custom_bundle(
            tools=["stl_decompose", "detect_peaks"],
            include_comparison=False,
        )

        tool_names = [t.name for t in bundle]
        assert "stl_decompose" in tool_names
        assert "detect_peaks" in tool_names

    def test_create_custom_bundle_by_category(self):
        """Test creating custom bundle by category."""
        from ts_agents.tools.bundles import create_custom_bundle
        from ts_agents.tools.registry import ToolCategory

        bundle = create_custom_bundle(
            categories=["decomposition"],
            include_comparison=False,
        )

        for tool in bundle:
            assert tool.category == ToolCategory.DECOMPOSITION

    def test_create_custom_bundle_with_max_cost(self):
        """Test creating custom bundle with max cost filter."""
        from ts_agents.tools.bundles import create_custom_bundle
        from ts_agents.tools.registry import ComputationalCost

        bundle = create_custom_bundle(
            categories=["forecasting"],
            max_cost=ComputationalCost.MEDIUM,
            include_comparison=False,
        )

        for tool in bundle:
            assert tool.cost in [ComputationalCost.LOW, ComputationalCost.MEDIUM]

    def test_create_custom_bundle_includes_comparison(self):
        """Test that custom bundle can include comparison tools."""
        from ts_agents.tools.bundles import create_custom_bundle

        bundle = create_custom_bundle(
            categories=["decomposition"],
            include_comparison=True,
        )

        tool_names = [t.name for t in bundle]
        # Should include at least one comparison tool
        assert any("compare" in name for name in tool_names)


class TestSubagentBundles:
    """Tests for subagent-specific bundles."""

    def test_get_decomposition_subagent_bundle(self):
        """Test getting decomposition subagent bundle."""
        from ts_agents.tools.bundles import get_subagent_bundle
        from ts_agents.tools.registry import ToolCategory

        tools = get_subagent_bundle("decomposition")

        assert len(tools) > 0
        # Most should be decomposition tools
        decomp_count = sum(1 for t in tools if t.category == ToolCategory.DECOMPOSITION)
        assert decomp_count >= len(tools) // 2

    def test_get_forecasting_subagent_bundle(self):
        """Test getting forecasting subagent bundle."""
        from ts_agents.tools.bundles import get_subagent_bundle

        tools = get_subagent_bundle("forecasting")

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert any("forecast" in name for name in tool_names)
        assert "forecast_seasonal_naive_with_data" in tool_names

    def test_get_patterns_subagent_bundle(self):
        """Test getting patterns subagent bundle."""
        from ts_agents.tools.bundles import get_subagent_bundle

        tools = get_subagent_bundle("patterns")

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert (
            "detect_peaks" in tool_names or
            "detect_peaks_with_data" in tool_names or
            "analyze_matrix_profile" in tool_names or
            "analyze_matrix_profile_with_data" in tool_names
        )

    def test_get_classification_subagent_bundle(self):
        """Test getting classification subagent bundle."""
        from ts_agents.tools.bundles import get_subagent_bundle

        tools = get_subagent_bundle("classification")

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert any("classify" in name for name in tool_names)

    def test_get_turbulence_subagent_bundle(self):
        """Test getting turbulence subagent bundle."""
        from ts_agents.tools.bundles import get_subagent_bundle

        tools = get_subagent_bundle("turbulence")

        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert (
            "compute_psd" in tool_names or
            "compute_psd_with_data" in tool_names or
            "compute_coherence" in tool_names or
            "compute_coherence_with_data" in tool_names
        )

    def test_invalid_subagent_name(self):
        """Test that invalid subagent name raises error."""
        from ts_agents.tools.bundles import get_subagent_bundle

        with pytest.raises(ValueError):
            get_subagent_bundle("nonexistent_subagent")


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_langchain_bundle_import(self):
        """Test that langchain bundle function exists."""
        from ts_agents.tools.bundles import get_langchain_bundle

        # Just test it's callable - actual wrapping tested separately
        assert callable(get_langchain_bundle)

    def test_get_deepagent_bundle_import(self):
        """Test that deepagent bundle function exists."""
        from ts_agents.tools.bundles import get_deepagent_bundle

        assert callable(get_deepagent_bundle)
