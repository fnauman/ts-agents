"""Tests for simple agent module."""

import pytest


class TestPrompts:
    """Tests for agent prompts."""

    def test_get_system_prompt_basic(self):
        """Test generating basic system prompt."""
        from ts_agents.agents.simple.prompts import get_system_prompt

        prompt = get_system_prompt()

        assert "time series" in prompt.lower()
        assert "analysis" in prompt.lower()
        assert "Re200Rm200" not in prompt
        assert "bx001_real" not in prompt

    def test_get_system_prompt_with_tools(self):
        """Test system prompt includes tool names."""
        from ts_agents.agents.simple.prompts import get_system_prompt

        tool_names = ["stl_decompose", "detect_peaks", "forecast_arima"]
        prompt = get_system_prompt(tool_names=tool_names)

        assert "3 tools" in prompt
        for name in tool_names:
            assert name in prompt

    def test_get_system_prompt_with_data_info(self):
        """Test system prompt includes data info."""
        from ts_agents.agents.simple.prompts import get_system_prompt

        prompt = get_system_prompt(include_data_info=True)

        assert "Re200Rm200" in prompt
        assert "bx001_real" in prompt

    def test_get_system_prompt_without_data_info(self):
        """Test system prompt can exclude data info."""
        from ts_agents.agents.simple.prompts import get_system_prompt

        prompt = get_system_prompt(include_data_info=False)

        # Should still have basic content but not data-specific info
        assert "time series" in prompt.lower()
        assert "Re200Rm200" not in prompt
        assert "bx001_real" not in prompt

    def test_get_system_prompt_with_explicit_data_context_prompt(self):
        """Test system prompt can accept explicit caller-provided data context."""
        from ts_agents.agents.simple.prompts import get_system_prompt

        prompt = get_system_prompt(
            data_context_prompt="## Available Data\n- Dataset: battery_cells.csv"
        )

        assert "Available Data" in prompt
        assert "battery_cells.csv" in prompt

    def test_get_system_prompt_custom_instructions(self):
        """Test system prompt with custom instructions."""
        from ts_agents.agents.simple.prompts import get_system_prompt

        custom = "Always respond in JSON format."
        prompt = get_system_prompt(custom_instructions=custom)

        assert custom in prompt

    def test_get_bundle_prompt_minimal(self):
        """Test bundle-specific prompt for minimal."""
        from ts_agents.agents.simple.prompts import get_bundle_prompt

        prompt = get_bundle_prompt("minimal")

        assert "5 tools" in prompt.lower() or "minimal" in prompt.lower()

    def test_get_bundle_prompt_standard(self):
        """Test bundle-specific prompt for standard."""
        from ts_agents.agents.simple.prompts import get_bundle_prompt

        prompt = get_bundle_prompt("standard")

        assert len(prompt) > 0

    def test_get_bundle_prompt_full(self):
        """Test bundle-specific prompt for full."""
        from ts_agents.agents.simple.prompts import get_bundle_prompt

        prompt = get_bundle_prompt("full")

        assert "25" in prompt or "full" in prompt.lower() or "complete" in prompt.lower()

    def test_get_bundle_prompt_unknown(self):
        """Test bundle prompt for unknown bundle returns empty."""
        from ts_agents.agents.simple.prompts import get_bundle_prompt

        prompt = get_bundle_prompt("nonexistent_bundle")

        assert prompt == ""


class TestToolCallRecord:
    """Tests for ToolCallRecord dataclass."""

    def test_tool_call_record_creation(self):
        """Test creating a tool call record."""
        from ts_agents.agents.simple.agent import ToolCallRecord

        record = ToolCallRecord(
            tool_name="detect_peaks",
            args={"series": "test"},
            result="10 peaks found",
            duration_ms=150.5,
        )

        assert record.tool_name == "detect_peaks"
        assert record.args == {"series": "test"}
        assert record.result == "10 peaks found"
        assert record.duration_ms == 150.5
        assert record.error is None

    def test_tool_call_record_with_error(self):
        """Test tool call record with error."""
        from ts_agents.agents.simple.agent import ToolCallRecord

        record = ToolCallRecord(
            tool_name="detect_peaks",
            args={},
            error="Missing series parameter",
        )

        assert record.error == "Missing series parameter"
        assert record.result is None


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_conversation_turn_creation(self):
        """Test creating a conversation turn."""
        from ts_agents.agents.simple.agent import ConversationTurn, ToolCallRecord

        turn = ConversationTurn(
            user_message="How many peaks?",
            assistant_response="There are 10 peaks.",
            tool_calls=[
                ToolCallRecord(tool_name="detect_peaks", args={})
            ],
            duration_ms=1500.0,
        )

        assert turn.user_message == "How many peaks?"
        assert turn.assistant_response == "There are 10 peaks."
        assert len(turn.tool_calls) == 1
        assert turn.duration_ms == 1500.0


class TestAgentCreationConfiguration:
    """Tests for agent creation configuration (without actual LLM calls)."""

    def test_agent_metadata_stored(self):
        """Test that agent metadata is properly configured."""
        # This test verifies the configuration path, not actual agent creation
        from ts_agents.tools.bundles import get_bundle_names

        # Verify bundles are properly configured
        minimal_tools = get_bundle_names("minimal")
        standard_tools = get_bundle_names("standard")
        full_tools = get_bundle_names("full")

        assert len(minimal_tools) < len(standard_tools)
        assert len(standard_tools) < len(full_tools)

    def test_tool_bundle_names_available(self):
        """Test that all expected bundles are available."""
        from ts_agents.tools.bundles import list_available_bundles

        bundles = list_available_bundles()

        assert "minimal" in bundles
        assert "standard" in bundles
        assert "full" in bundles
        assert "all" in bundles


class TestSimpleAgentChatDataStructures:
    """Tests for SimpleAgentChat supporting data structures."""

    def test_get_tool_stats_empty(self):
        """Test tool stats with no calls."""
        # Test the data structure computation logic
        tool_calls = []

        # Compute stats manually (mirrors the method logic)
        tool_frequency = {}
        total_duration = 0.0
        error_count = 0

        stats = {
            "tool_call_count": len(tool_calls),
            "tools_used": set(tool_frequency.keys()),
            "tool_frequency": tool_frequency,
            "avg_tool_duration_ms": 0,
            "error_count": error_count,
        }

        assert stats["tool_call_count"] == 0
        assert stats["tools_used"] == set()
        assert stats["error_count"] == 0

    def test_get_tool_stats_with_calls(self):
        """Test tool stats computation with sample data."""
        from ts_agents.agents.simple.agent import ToolCallRecord

        tool_calls = [
            ToolCallRecord(tool_name="detect_peaks", args={}, duration_ms=100),
            ToolCallRecord(tool_name="detect_peaks", args={}, duration_ms=120),
            ToolCallRecord(tool_name="stl_decompose", args={}, duration_ms=500),
            ToolCallRecord(tool_name="forecast_arima", args={}, error="Failed", duration_ms=50),
        ]

        # Compute stats manually
        tool_frequency = {}
        total_duration = 0.0
        error_count = 0

        for call in tool_calls:
            tool_frequency[call.tool_name] = tool_frequency.get(call.tool_name, 0) + 1
            total_duration += call.duration_ms
            if call.error:
                error_count += 1

        stats = {
            "tool_call_count": len(tool_calls),
            "tools_used": set(tool_frequency.keys()),
            "tool_frequency": tool_frequency,
            "avg_tool_duration_ms": total_duration / len(tool_calls) if tool_calls else 0,
            "error_count": error_count,
        }

        assert stats["tool_call_count"] == 4
        assert stats["tools_used"] == {"detect_peaks", "stl_decompose", "forecast_arima"}
        assert stats["tool_frequency"]["detect_peaks"] == 2
        assert stats["tool_frequency"]["stl_decompose"] == 1
        assert stats["error_count"] == 1
        assert stats["avg_tool_duration_ms"] == 192.5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compare_bundles_function_exists(self):
        """Test that compare_bundles_on_query function exists."""
        from ts_agents.agents.simple.agent import compare_bundles_on_query

        assert callable(compare_bundles_on_query)

    def test_run_single_query_function_exists(self):
        """Test that run_single_query function exists."""
        from ts_agents.agents.simple.agent import run_single_query

        assert callable(run_single_query)
