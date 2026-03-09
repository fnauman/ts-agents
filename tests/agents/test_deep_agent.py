"""Tests for deep agent module."""

import pytest


class TestSubagentDefinitions:
    """Tests for subagent configurations."""

    def test_decomposition_subagent_structure(self):
        """Test decomposition subagent has required fields."""
        from ts_agents.agents.deep.subagents import DECOMPOSITION_SUBAGENT

        assert "name" in DECOMPOSITION_SUBAGENT
        assert "description" in DECOMPOSITION_SUBAGENT
        assert "system_prompt" in DECOMPOSITION_SUBAGENT

        assert DECOMPOSITION_SUBAGENT["name"] == "decomposition-agent"
        assert "decomposition" in DECOMPOSITION_SUBAGENT["description"].lower()
        assert len(DECOMPOSITION_SUBAGENT["system_prompt"]) > 100

    def test_forecasting_subagent_structure(self):
        """Test forecasting subagent has required fields."""
        from ts_agents.agents.deep.subagents import FORECASTING_SUBAGENT

        assert "name" in FORECASTING_SUBAGENT
        assert "description" in FORECASTING_SUBAGENT
        assert "system_prompt" in FORECASTING_SUBAGENT

        assert FORECASTING_SUBAGENT["name"] == "forecasting-agent"
        assert "forecast" in FORECASTING_SUBAGENT["description"].lower()

    def test_patterns_subagent_structure(self):
        """Test patterns subagent has required fields."""
        from ts_agents.agents.deep.subagents import PATTERNS_SUBAGENT

        assert "name" in PATTERNS_SUBAGENT
        assert "description" in PATTERNS_SUBAGENT
        assert "system_prompt" in PATTERNS_SUBAGENT

        assert PATTERNS_SUBAGENT["name"] == "patterns-agent"
        assert "pattern" in PATTERNS_SUBAGENT["description"].lower()

    def test_classification_subagent_structure(self):
        """Test classification subagent has required fields."""
        from ts_agents.agents.deep.subagents import CLASSIFICATION_SUBAGENT

        assert "name" in CLASSIFICATION_SUBAGENT
        assert "description" in CLASSIFICATION_SUBAGENT
        assert "system_prompt" in CLASSIFICATION_SUBAGENT

        assert CLASSIFICATION_SUBAGENT["name"] == "classification-agent"
        assert "classif" in CLASSIFICATION_SUBAGENT["description"].lower()

    def test_turbulence_subagent_structure(self):
        """Test turbulence subagent has required fields."""
        from ts_agents.agents.deep.subagents import TURBULENCE_SUBAGENT

        assert "name" in TURBULENCE_SUBAGENT
        assert "description" in TURBULENCE_SUBAGENT
        assert "system_prompt" in TURBULENCE_SUBAGENT

        assert TURBULENCE_SUBAGENT["name"] == "turbulence-agent"
        assert "turbulence" in TURBULENCE_SUBAGENT["description"].lower() or \
               "cfd" in TURBULENCE_SUBAGENT["description"].lower()


class TestSubagentTools:
    """Tests for subagent tool retrieval."""

    def test_decomposition_tools_available(self):
        """Test decomposition subagent tools can be retrieved."""
        from ts_agents.agents.deep.subagents.decomposition import get_decomposition_tools

        tools = get_decomposition_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check tool structure
        tool = tools[0]
        assert "name" in tool
        assert "description" in tool

    def test_forecasting_tools_available(self):
        """Test forecasting subagent tools can be retrieved."""
        from ts_agents.agents.deep.subagents.forecasting import get_forecasting_tools

        tools = get_forecasting_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_patterns_tools_available(self):
        """Test patterns subagent tools can be retrieved."""
        from ts_agents.agents.deep.subagents.patterns import get_patterns_tools

        tools = get_patterns_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_classification_tools_available(self):
        """Test classification subagent tools can be retrieved."""
        from ts_agents.agents.deep.subagents.classification import get_classification_tools

        tools = get_classification_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_turbulence_tools_available(self):
        """Test turbulence subagent tools can be retrieved."""
        from ts_agents.agents.deep.subagents.turbulence import get_turbulence_tools

        tools = get_turbulence_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0


class TestSubagentCall:
    """Tests for SubagentCall dataclass."""

    def test_subagent_call_creation(self):
        """Test creating a subagent call record."""
        from ts_agents.agents.deep.orchestrator import SubagentCall

        call = SubagentCall(
            subagent_name="decomposition-agent",
            query="Decompose the signal",
            response="Decomposition complete. Trend: ...",
            duration_ms=2500.0,
        )

        assert call.subagent_name == "decomposition-agent"
        assert call.query == "Decompose the signal"
        assert "Decomposition complete" in call.response
        assert call.duration_ms == 2500.0
        assert call.timestamp is not None


class TestDeepAgentTurn:
    """Tests for DeepAgentTurn dataclass."""

    def test_deep_agent_turn_creation(self):
        """Test creating a deep agent turn record."""
        from ts_agents.agents.deep.orchestrator import DeepAgentTurn, SubagentCall

        turn = DeepAgentTurn(
            user_message="Analyze the spectral properties",
            assistant_response="The spectral analysis shows...",
            subagent_calls=[
                SubagentCall(
                    subagent_name="turbulence-agent",
                    query="Compute PSD",
                    response="PSD computed",
                    duration_ms=500.0,
                )
            ],
            duration_ms=3000.0,
        )

        assert turn.user_message == "Analyze the spectral properties"
        assert turn.assistant_response == "The spectral analysis shows..."
        assert len(turn.subagent_calls) == 1
        assert turn.subagent_calls[0].subagent_name == "turbulence-agent"
        assert turn.duration_ms == 3000.0
        assert turn.required_approval == False

    def test_deep_agent_turn_with_approval(self):
        """Test deep agent turn that required approval."""
        from ts_agents.agents.deep.orchestrator import DeepAgentTurn

        turn = DeepAgentTurn(
            user_message="Run HC2 classifier",
            assistant_response="Classification complete",
            required_approval=True,
            duration_ms=60000.0,  # Long running
        )

        assert turn.required_approval == True


class TestOrchestratorConfiguration:
    """Tests for orchestrator configuration (without actual LLM calls)."""

    def test_get_all_subagents(self):
        """Test retrieving all subagent configurations."""
        from ts_agents.agents.deep.orchestrator import get_all_subagents

        subagents = get_all_subagents()

        assert isinstance(subagents, list)
        assert len(subagents) == 5  # 5 specialized subagents

        # Verify each has required fields and tools
        for subagent in subagents:
            assert "name" in subagent
            assert "description" in subagent
            assert "system_prompt" in subagent
            assert "tools" in subagent
            assert len(subagent["tools"]) > 0

    def test_subagent_names(self):
        """Test that all expected subagent names are present."""
        from ts_agents.agents.deep.orchestrator import get_all_subagents

        subagents = get_all_subagents()
        names = [s["name"] for s in subagents]

        assert "decomposition-agent" in names
        assert "forecasting-agent" in names
        assert "patterns-agent" in names
        assert "classification-agent" in names
        assert "turbulence-agent" in names

    def test_create_interrupt_config_enabled(self):
        """Test interrupt config when approval is enabled."""
        from ts_agents.agents.deep.orchestrator import create_interrupt_config

        config = create_interrupt_config(enable_approval=True)

        # Should have config for VERY_HIGH cost tools
        if config:  # Only if there are expensive tools
            assert isinstance(config, dict)
            for tool_name, should_interrupt in config.items():
                assert should_interrupt == True

    def test_create_interrupt_config_disabled(self):
        """Test interrupt config when approval is disabled."""
        from ts_agents.agents.deep.orchestrator import create_interrupt_config

        config = create_interrupt_config(enable_approval=False)

        assert config is None

    def test_get_expensive_tool_names(self):
        """Test retrieving expensive tool names."""
        from ts_agents.agents.deep.orchestrator import get_expensive_tool_names

        expensive = get_expensive_tool_names()

        assert isinstance(expensive, list)
        # Should include HC2 classifier (VERY_HIGH cost)
        assert "hivecote_classify" in expensive

    def test_create_deep_agent_does_not_require_langchain_anthropic(self, monkeypatch):
        """Deep agent creation should not fail if Anthropic package is absent."""
        import sys
        import ts_agents.agents.deep.orchestrator as orchestrator

        monkeypatch.delitem(sys.modules, "langchain_anthropic", raising=False)
        monkeypatch.setattr(orchestrator, "get_bundle", lambda name: [])
        monkeypatch.setattr(orchestrator, "wrap_tools_for_deepagent", lambda bundle: [])
        monkeypatch.setattr(orchestrator, "get_all_subagents", lambda: [])
        monkeypatch.setattr(orchestrator, "_create_with_deepagents", lambda **kwargs: {"ok": True})

        agent = orchestrator.create_deep_agent(
            model_name="gpt-5-mini",
            enable_approval=False,
            enable_logging=False,
        )
        assert agent == {"ok": True}


class TestUtilityFunctions:
    """Tests for deep agent utility functions."""

    def test_list_subagents(self):
        """Test listing subagents."""
        from ts_agents.agents.deep import list_subagents

        subagents = list_subagents()

        assert isinstance(subagents, list)
        assert len(subagents) == 5

        for s in subagents:
            assert "name" in s
            assert "description" in s

    def test_get_expensive_tools(self):
        """Test getting expensive tools info."""
        from ts_agents.agents.deep import get_expensive_tools

        tools = get_expensive_tools()

        assert isinstance(tools, list)
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "category" in tool

    def test_run_with_approval_function_exists(self):
        """Test that run_with_approval function exists."""
        from ts_agents.agents.deep import run_with_approval

        assert callable(run_with_approval)


class TestDeepAgentChatDataStructures:
    """Tests for DeepAgentChat supporting data structures."""

    def test_get_stats_computation(self):
        """Test stats computation logic."""
        from ts_agents.agents.deep.orchestrator import SubagentCall

        # Sample data
        subagent_calls = [
            SubagentCall(subagent_name="decomposition-agent", query="q1",
                        response="r1", duration_ms=100),
            SubagentCall(subagent_name="decomposition-agent", query="q2",
                        response="r2", duration_ms=150),
            SubagentCall(subagent_name="turbulence-agent", query="q3",
                        response="r3", duration_ms=200),
        ]

        # Compute stats manually (mirrors the method logic)
        subagent_frequency = {}
        for call in subagent_calls:
            subagent_frequency[call.subagent_name] = (
                subagent_frequency.get(call.subagent_name, 0) + 1
            )

        stats = {
            "subagent_calls": len(subagent_calls),
            "subagent_frequency": subagent_frequency,
        }

        assert stats["subagent_calls"] == 3
        assert stats["subagent_frequency"]["decomposition-agent"] == 2
        assert stats["subagent_frequency"]["turbulence-agent"] == 1


class TestSystemPrompts:
    """Tests for system prompts."""

    def test_orchestrator_system_prompt(self):
        """Test orchestrator system prompt content."""
        from ts_agents.agents.deep.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

        prompt = ORCHESTRATOR_SYSTEM_PROMPT

        # Should mention sub-agents
        assert "decomposition-agent" in prompt
        assert "forecasting-agent" in prompt
        assert "patterns-agent" in prompt
        assert "classification-agent" in prompt
        assert "turbulence-agent" in prompt

        # Should mention delegation
        assert "delegate" in prompt.lower()

        # Should mention cost awareness
        assert "cost" in prompt.lower()

    def test_decomposition_system_prompt(self):
        """Test decomposition agent system prompt."""
        from ts_agents.agents.deep.subagents.decomposition import DECOMPOSITION_SYSTEM_PROMPT

        prompt = DECOMPOSITION_SYSTEM_PROMPT

        # Should mention methods
        assert "STL" in prompt
        assert "MSTL" in prompt
        assert "HP" in prompt or "Hodrick" in prompt
        assert "Holt-Winters" in prompt

    def test_forecasting_system_prompt(self):
        """Test forecasting agent system prompt."""
        from ts_agents.agents.deep.subagents.forecasting import FORECASTING_SYSTEM_PROMPT

        prompt = FORECASTING_SYSTEM_PROMPT

        # Should mention methods
        assert "ARIMA" in prompt
        assert "ETS" in prompt
        assert "Theta" in prompt

    def test_patterns_system_prompt(self):
        """Test patterns agent system prompt."""
        from ts_agents.agents.deep.subagents.patterns import PATTERNS_SYSTEM_PROMPT

        prompt = PATTERNS_SYSTEM_PROMPT

        # Should mention key concepts
        assert "motif" in prompt.lower()
        assert "discord" in prompt.lower() or "anomal" in prompt.lower()
        assert "peak" in prompt.lower()

    def test_classification_system_prompt(self):
        """Test classification agent system prompt."""
        from ts_agents.agents.deep.subagents.classification import CLASSIFICATION_SYSTEM_PROMPT

        prompt = CLASSIFICATION_SYSTEM_PROMPT

        # Should mention classifiers
        assert "DTW" in prompt or "KNN" in prompt
        assert "ROCKET" in prompt
        assert "HIVE-COTE" in prompt or "HC2" in prompt

        # Should mention data format
        assert "3D" in prompt or "n_samples" in prompt

    def test_turbulence_system_prompt(self):
        """Test turbulence agent system prompt."""
        from ts_agents.agents.deep.subagents.turbulence import TURBULENCE_SYSTEM_PROMPT

        prompt = TURBULENCE_SYSTEM_PROMPT

        # Should mention CFD concepts
        assert "Reynolds" in prompt or "Re" in prompt
        assert "spectral" in prompt.lower()
        assert "PSD" in prompt or "power spectral" in prompt.lower()

        # Should mention Kolmogorov
        assert "Kolmogorov" in prompt or "-5/3" in prompt


class TestImports:
    """Tests for module imports."""

    def test_deep_agent_imports(self):
        """Test all deep agent imports work."""
        from ts_agents.agents.deep import (
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

        # All should be importable
        assert callable(create_deep_agent)
        assert DeepAgentChat is not None
        assert DeepAgentTurn is not None
        assert SubagentCall is not None

    def test_subagent_imports(self):
        """Test subagent imports work."""
        from ts_agents.agents.deep import (
            DECOMPOSITION_SUBAGENT,
            FORECASTING_SUBAGENT,
            PATTERNS_SUBAGENT,
            CLASSIFICATION_SUBAGENT,
            TURBULENCE_SUBAGENT,
        )

        assert isinstance(DECOMPOSITION_SUBAGENT, dict)
        assert isinstance(FORECASTING_SUBAGENT, dict)
        assert isinstance(PATTERNS_SUBAGENT, dict)
        assert isinstance(CLASSIFICATION_SUBAGENT, dict)
        assert isinstance(TURBULENCE_SUBAGENT, dict)

    def test_agents_package_exports_deep(self):
        """Test main agents package exports deep agent."""
        from ts_agents.agents import create_deep_agent, DeepAgentChat, list_subagents

        assert callable(create_deep_agent)
        assert DeepAgentChat is not None
        assert callable(list_subagents)
