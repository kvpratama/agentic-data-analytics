"""Tests for the write_todos tool and todos state field."""

from __future__ import annotations

from langgraph.types import Command

from ada_lg.lg.todos_tool import AnalyticsState, todos_reducer, write_todos


def test_todos_reducer_replaces_existing_list() -> None:
    """A new todos list replaces the old list."""
    old = [{"content": "step 1", "status": "completed"}]
    new = [
        {"content": "step 2", "status": "in_progress"},
        {"content": "step 3", "status": "pending"},
    ]
    assert todos_reducer(old, new) == new


def test_todos_reducer_keeps_old_when_new_is_none() -> None:
    """A None update leaves the current todos in place."""
    old = [{"content": "x", "status": "pending"}]
    assert todos_reducer(old, None) == old


def test_state_schema_includes_todos_field() -> None:
    """AnalyticsState includes the planning field."""
    assert "todos" in AnalyticsState.__annotations__


async def test_write_todos_returns_summary() -> None:
    """write_todos returns a ToolMessage summary via Command."""
    result = await write_todos.ainvoke(
        {
            "type": "tool_call",
            "name": "write_todos",
            "id": "abc",
            "args": {
                "todos": [
                    {"content": "Profile the data", "status": "in_progress"},
                    {"content": "Clean the data", "status": "pending"},
                ],
            },
        }
    )
    assert isinstance(result, Command)
    assert "2" in result.update["messages"][0].content


async def test_write_todos_returns_command_updating_state() -> None:
    """write_todos updates the todos state field."""
    result = await write_todos.ainvoke(
        {
            "type": "tool_call",
            "name": "write_todos",
            "id": "call_123",
            "args": {
                "todos": [{"content": "X", "status": "pending"}],
            },
        }
    )
    assert isinstance(result, Command)
    assert result.update["todos"] == [{"content": "X", "status": "pending"}]
    assert result.update["messages"][0].tool_call_id == "call_123"
