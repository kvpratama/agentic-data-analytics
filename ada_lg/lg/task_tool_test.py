"""Tests for the orchestrator's task delegation tool."""

from __future__ import annotations

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage

from ada_lg.lg.task_tool import build_task_tool


def _fake_graph(final_text: str) -> AsyncMock:
    """Build a fake compiled graph with an async invocation."""
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "messages": [
            HumanMessage("input"),
            AIMessage(final_text),
        ]
    }
    return graph


async def test_task_invokes_named_subagent_with_instruction() -> None:
    """task routes to the named subagent with a fresh user message."""
    profiler = _fake_graph("profile written to /workspace/profile.json")
    cleaner = _fake_graph("cleaned")
    task = build_task_tool({"profiler": profiler, "cleaner": cleaner})

    result = await task.ainvoke({"agent": "profiler", "instruction": "Profile dataset.csv"})

    profiler.ainvoke.assert_awaited_once()
    cleaner.ainvoke.assert_not_awaited()
    sent = profiler.ainvoke.await_args.args[0]
    assert sent["messages"][0].content == "Profile dataset.csv"
    assert "profile written" in result


async def test_task_errors_on_unknown_agent() -> None:
    """Unknown subagent names return a model-readable error."""
    task = build_task_tool({"profiler": _fake_graph("x")})
    result = await task.ainvoke({"agent": "nope", "instruction": "do thing"})
    assert "Error" in result
    assert "nope" in result
