"""Tests for build_subagent and build_orchestrator wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deepagents.backends import FilesystemBackend

from ada_lg.lg.agent_factory import build_orchestrator, build_subagent
from ada_lg.lg.todos_tool import AnalyticsState


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    """Create a small skill tree for factory tests."""
    base = tmp_path / "profiler_skills" / "x"
    base.mkdir(parents=True)
    (base / "SKILL.md").write_text("---\nname: x\ndescription: dummy\n---\nBody.\n")
    return tmp_path


def _backend(root: Path) -> FilesystemBackend:
    """Return a filesystem backend rooted at test skills."""
    return FilesystemBackend(root_dir=str(root), virtual_mode=True)


def test_build_subagent_registers_expected_tool_names(skills_root: Path) -> None:
    """Subagents receive FS, skill, and todos tools, but not task."""
    backend = _backend(skills_root)
    graph = MagicMock(name="graph")

    with patch("ada_lg.lg.agent_factory.create_agent", return_value=graph) as create_agent:
        result = build_subagent(
            name="profiler",
            system_prompt="You are profiler.",
            model=MagicMock(name="model"),
            middleware=[],
            backend=backend,
            skill_dir="/profiler_skills/",
        )

    assert result is graph
    kwargs = create_agent.call_args.kwargs
    tool_names = {tool.name for tool in kwargs["tools"]}
    assert {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "read_skill",
        "write_todos",
    }.issubset(tool_names)
    assert "task" not in tool_names
    assert kwargs["state_schema"] is AnalyticsState
    assert kwargs["name"] == "profiler"
    assert "x: dummy" in kwargs["system_prompt"]


def test_build_orchestrator_includes_task_tool_and_checkpointer(skills_root: Path) -> None:
    """The orchestrator receives task and forwards the checkpointer."""
    backend = _backend(skills_root)
    graph = MagicMock(name="graph")
    checkpointer = MagicMock(name="checkpointer")

    with patch("ada_lg.lg.agent_factory.create_agent", return_value=graph) as create_agent:
        result = build_orchestrator(
            system_prompt="You orchestrate.",
            model=MagicMock(name="model"),
            middleware=[],
            backend=backend,
            subagents={"profiler": MagicMock(name="profiler-graph")},
            skill_dir="/profiler_skills/",
            checkpointer=checkpointer,
        )

    assert result is graph
    kwargs = create_agent.call_args.kwargs
    tool_names = {tool.name for tool in kwargs["tools"]}
    assert "task" in tool_names
    assert kwargs["checkpointer"] is checkpointer
    assert "profiler" in kwargs["system_prompt"]


async def test_write_tools_are_guarded(skills_root: Path) -> None:
    """Factory wraps write/edit tools with the skills write-deny guard."""
    backend = _backend(skills_root)
    with patch("ada_lg.lg.agent_factory.create_agent", return_value=MagicMock()) as create_agent:
        build_subagent(
            name="profiler",
            system_prompt="You are profiler.",
            model=MagicMock(name="model"),
            middleware=[],
            backend=backend,
            skill_dir="/profiler_skills/",
        )

    tools = {tool.name: tool for tool in create_agent.call_args.kwargs["tools"]}
    result = await tools["write_file"].ainvoke({"path": "/skills/x/SKILL.md", "content": "bad"})
    assert "denied" in result.lower()
