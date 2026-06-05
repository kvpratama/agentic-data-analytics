"""Orchestrator-only subagent delegation tool."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from langchain.tools import BaseTool, tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph


def _message_content_to_text(content: object) -> str:
    """Convert AIMessage content blocks to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text_block = cast("Mapping[str, object]", block)
                parts.append(str(text_block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)


def build_task_tool(subagents: Mapping[str, CompiledStateGraph]) -> BaseTool:
    """Build a task(agent, instruction) tool over a subagent registry.

    Args:
        subagents: Mapping of subagent names to compiled graphs.

    Returns:
        LangChain tool that invokes the requested subagent.
    """
    registry = dict(subagents)
    available = ", ".join(registry) or "(none)"

    @tool
    async def task(agent: str, instruction: str) -> str:
        """Delegate a self-contained task to a named subagent."""
        graph = registry.get(agent)
        if graph is None:
            return f"Error: unknown agent {agent!r}. Available: {available}"
        state = await graph.ainvoke({"messages": [HumanMessage(instruction)]})
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                return _message_content_to_text(message.content)
        return "(subagent returned no AI message)"

    return task
