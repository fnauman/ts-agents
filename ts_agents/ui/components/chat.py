"""Chat component for agent interaction.

This component provides a chat interface for interacting with:
- Simple agent (LangChain-based)
- Deep agent (multi-agent orchestration)
"""

import gradio as gr
from typing import List, Tuple, Optional, Any
import base64
import re
from io import BytesIO

from ..state import UIState


# Agent type constants
AGENT_SIMPLE = "Simple Agent (LangChain)"
AGENT_DEEP = "Deep Agent (Multi-agent)"


def extract_image_from_response(response: str) -> Tuple[str, Optional[Any]]:
    """Extract base64 image data from response if present.

    Returns:
        Tuple of (cleaned response text, PIL Image or None)
    """
    # Look for [IMAGE_DATA:base64string] pattern
    pattern = r'\[IMAGE_DATA:([A-Za-z0-9+/=]+)\]'
    match = re.search(pattern, response)

    if match:
        try:
            from PIL import Image
            img_data = base64.b64decode(match.group(1))
            img = Image.open(BytesIO(img_data))
            cleaned_response = re.sub(pattern, '[Image generated - see below]', response)
            return cleaned_response, img
        except Exception:
            pass

    return response, None


def create_chat_tab(state: gr.State, agent_type: str = "simple"):
    """Create the agent chat tab.

    Parameters
    ----------
    state : gr.State
        Gradio state object containing UIState
    agent_type : str
        Default agent type: "simple" or "deep"
    """
    # Example queries
    example_queries = [
        # Data exploration
        "What data is available in the dataset?",
        "List all the runs and variables I can analyze",
        # Peak detection
        "How many peaks are in bx001_real for Re200Rm200?",
        "Find peaks in vy001_imag for Re175Rm175",
        # Decomposition
        "Decompose bx001_real for Re200Rm200 and show me the trend",
        "Compare all decomposition methods on by001_real",
        # Forecasting
        "Forecast bx001_real for Re200Rm200 for the next 20 time steps",
        "Compare forecasting methods for by001_real",
        # Pattern analysis
        "Find motifs in bx001_real for Re200Rm200",
        "Show me a recurrence plot for by001_real",
        "Detect changepoints in vx001_imag for Re150Rm150",
        # Spectral analysis
        "Compute the power spectrum of bx001_real",
        "What is the dominant frequency in by001_real?",
        # Comparative queries
        "Compare the peak counts across all runs for bx001_real",
    ]

    gr.Markdown("""
    ## Agent Chat

    Ask questions about the time series data using natural language.
    The agent can perform analysis, generate plots, and compare methods.
    """)

    with gr.Row():
        agent_selector = gr.Dropdown(
            choices=[AGENT_SIMPLE, AGENT_DEEP],
            value=AGENT_SIMPLE if agent_type == "simple" else AGENT_DEEP,
            label="Agent Type",
            info="Simple: Single agent with all tools. Deep: Multi-agent with specialists.",
            scale=2
        )
        tool_bundle = gr.Dropdown(
            choices=["demo", "minimal", "standard", "full"],
            value="standard",
            label="Tool Bundle (Simple Agent)",
            info="Number of tools available to the agent",
            scale=1
        )

    chatbot = gr.Chatbot(
        label="Conversation",
        height=450,
        # Gradio 6.0: messages format is default and only option
    )

    image_output = gr.Image(
        label="Generated Plot",
        visible=True,
        type="pil"
    )

    # Tool calls log - collapsible section to show what tools the model called
    with gr.Accordion("Tool Calls Log", open=False):
        tool_calls_log = gr.Markdown(
            value="*No tool calls yet. Tool calls will appear here after you send a message.*",
            label="Tool Calls"
        )

    with gr.Row():
        msg_input = gr.Textbox(
            label="Your question",
            placeholder="Ask about the time series data...",
            scale=4,
            lines=2
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear Chat")
        export_btn = gr.Button("Export Session")

    gr.Markdown("### Example Queries")
    gr.Examples(
        examples=example_queries,
        inputs=msg_input,
        label="Click an example to try it:",
    )

    # State for tracking the current agent
    current_agent_state = gr.State({"agent": None, "type": None, "bundle": None})

    def get_or_create_agent(agent_state, agent_type_str, bundle):
        """Get existing agent or create a new one if config changed."""
        if (agent_state["agent"] is not None and
            agent_state["type"] == agent_type_str and
            agent_state["bundle"] == bundle):
            return agent_state["agent"], agent_state

        # Create new agent with user-friendly error handling for missing dependencies
        try:
            if agent_type_str == AGENT_SIMPLE:
                try:
                    from ...agents.simple import SimpleAgentChat
                except ImportError as e:
                    error_msg = (
                        "Simple Agent requires LangChain packages. "
                        "Please install: pip install langchain langchain-openai"
                    )
                    print(f"Import error: {e}")
                    return None, {**agent_state, "error": error_msg}
                agent = SimpleAgentChat(tool_bundle=bundle)
            else:
                # Deep agent
                try:
                    from ...agents.deep import DeepAgentChat
                except ImportError as e:
                    error_msg = (
                        "Deep Agent requires LangChain packages. "
                        "Please install: pip install langchain langchain-openai langchain-anthropic"
                    )
                    print(f"Import error: {e}")
                    return None, {**agent_state, "error": error_msg}
                agent = DeepAgentChat()

            new_state = {"agent": agent, "type": agent_type_str, "bundle": bundle}
            return agent, new_state
        except Exception as e:
            print(f"Error creating agent: {e}")
            return None, {**agent_state, "error": str(e)}

    def format_tool_calls_log(agent) -> str:
        """Format tool calls from the agent for display."""
        if agent is None:
            return "*No tool calls yet.*"

        try:
            # Get tool calls from agent's internal log
            tool_calls = getattr(agent, '_tool_calls', [])
            if not tool_calls:
                return "*No tool calls recorded for this session.*"

            lines = ["### Recent Tool Calls\n"]
            for i, call in enumerate(tool_calls[-10:], 1):  # Show last 10 calls
                status_emoji = "✅" if call.error is None else "❌"
                lines.append(f"**{i}. {status_emoji} `{call.tool_name}`**")
                lines.append(f"   - Args: `{call.args}`")
                if call.result:
                    # Truncate long results
                    result_preview = call.result[:200] + "..." if len(call.result) > 200 else call.result
                    lines.append(f"   - Result: {result_preview}")
                if call.error:
                    lines.append(f"   - Error: {call.error}")
                lines.append(f"   - Duration: {call.duration_ms:.1f}ms")
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            return f"*Error formatting tool calls: {e}*"

    def user_submit(user_message, history, ui_state, agent_state, agent_type_str, bundle):
        """Handle user message submission."""
        if not user_message.strip():
            return "", history, None, ui_state, agent_state, "*No tool calls yet.*"

        # Add user message to history
        history = history + [{"role": "user", "content": user_message}]

        # Get or create agent
        agent, new_agent_state = get_or_create_agent(agent_state, agent_type_str, bundle)

        if agent is None:
            # Check for specific error message from agent creation
            error_msg = new_agent_state.get("error",
                "Failed to initialize agent. Please check your configuration and ensure "
                "required packages are installed (langchain, langchain-openai).")
            history = history + [{"role": "assistant", "content": f"⚠️ {error_msg}"}]
            return "", history, None, ui_state, new_agent_state, "*Agent initialization failed.*"

        # Get agent response
        try:
            response = agent.chat(user_message)

            # Check for image in response
            cleaned_response, img = extract_image_from_response(response)

            history = history + [{"role": "assistant", "content": cleaned_response}]

            # Update UI state
            if ui_state is not None:
                ui_state.add_chat_message("user", user_message)
                ui_state.add_chat_message("assistant", cleaned_response)

            # Format tool calls log
            tool_log = format_tool_calls_log(agent)

            return "", history, img, ui_state, new_agent_state, tool_log

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history = history + [{"role": "assistant", "content": error_msg}]
            tool_log = format_tool_calls_log(agent)
            return "", history, None, ui_state, new_agent_state, tool_log

    def clear_chat_handler(ui_state, agent_state):
        """Clear chat history."""
        if agent_state["agent"] is not None:
            try:
                agent_state["agent"].reset()
            except Exception:
                pass

        if ui_state is not None:
            ui_state.clear_chat()

        return [], None, ui_state, "*No tool calls yet. Tool calls will appear here after you send a message.*"

    def switch_agent(agent_type_str, agent_state):
        """Switch to a different agent type."""
        # Reset agent state to force recreation
        return [], None, {"agent": None, "type": None, "bundle": None}, "*No tool calls yet. Tool calls will appear here after you send a message.*"

    def export_session_handler(agent_state):
        """Export the current session."""
        if agent_state["agent"] is None:
            return gr.Info("No active session to export")

        try:
            from datetime import datetime
            filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            agent_state["agent"].export_session(filename)
            return gr.Info(f"Session exported to {filename}")
        except Exception as e:
            return gr.Warning(f"Export failed: {e}")

    # Connect events
    submit_btn.click(
        user_submit,
        inputs=[msg_input, chatbot, state, current_agent_state, agent_selector, tool_bundle],
        outputs=[msg_input, chatbot, image_output, state, current_agent_state, tool_calls_log]
    )

    msg_input.submit(
        user_submit,
        inputs=[msg_input, chatbot, state, current_agent_state, agent_selector, tool_bundle],
        outputs=[msg_input, chatbot, image_output, state, current_agent_state, tool_calls_log]
    )

    clear_btn.click(
        clear_chat_handler,
        inputs=[state, current_agent_state],
        outputs=[chatbot, image_output, state, tool_calls_log]
    )

    agent_selector.change(
        switch_agent,
        inputs=[agent_selector, current_agent_state],
        outputs=[chatbot, image_output, current_agent_state, tool_calls_log]
    )

    export_btn.click(
        export_session_handler,
        inputs=[current_agent_state],
        outputs=[]
    )

    # Show/hide tool bundle selector based on agent type
    def update_tool_bundle_visibility(agent_type_str):
        return gr.update(visible=(agent_type_str == AGENT_SIMPLE))

    agent_selector.change(
        update_tool_bundle_visibility,
        inputs=[agent_selector],
        outputs=[tool_bundle]
    )
