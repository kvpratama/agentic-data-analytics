"""Planning state field and write_todos tool."""

from __future__ import annotations

from typing import Annotated

from langchain.agents import AgentState
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command


def todos_reducer(existing: list[dict] | None, update: list[dict] | None) -> list[dict]:
    """Return the replacement todos list, or preserve the existing list.

    Args:
        existing: Current todos.
        update: New todos, or None to leave state unchanged.

    Returns:
        The active todos list.
    """
    if update is None:
        return existing or []
    return update


class AnalyticsState(AgentState):
    """Agent state extended with a todos planning field."""

    todos: Annotated[list[dict], todos_reducer]


@tool
async def write_todos(
    todos: list[dict],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Record or update the agent's plan as a list of todo items."""
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(content=f"Recorded {len(todos)} todo(s).", tool_call_id=tool_call_id)
            ],
        }
    )
