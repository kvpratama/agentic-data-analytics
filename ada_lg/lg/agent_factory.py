"""Compose LangChain create_agent graphs for ADA's orchestrator and subagents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from deepagents.backends.protocol import BackendProtocol
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from ada_lg.lg.fs_tools import build_fs_tools
from ada_lg.lg.permissions import wrap_write_deny
from ada_lg.lg.skills import build_skills
from ada_lg.lg.task_tool import build_task_tool
from ada_lg.lg.todos_tool import AnalyticsState, write_todos

_DENIED_WRITE_PREFIXES = ("/skills/",)


def _guard_writes(tools: list[BaseTool]) -> list[BaseTool]:
    """Wrap write-capable filesystem tools with the /skills/ write deny guard."""
    guarded: list[BaseTool] = []
    for item in tools:
        if item.name in {"write_file", "edit_file"}:
            guarded.append(wrap_write_deny(item, prefixes=_DENIED_WRITE_PREFIXES))
        else:
            guarded.append(item)
    return guarded


def build_subagent(
    *,
    name: str,
    system_prompt: str,
    model: BaseChatModel,
    middleware: Sequence[AgentMiddleware],
    backend: BackendProtocol,
    skill_dir: str,
) -> CompiledStateGraph:
    """Build a compiled subagent graph.

    Args:
        name: Subagent name.
        system_prompt: Subagent instructions.
        model: Chat model.
        middleware: Retry/fallback middleware.
        backend: Shared filesystem/sandbox backend.
        skill_dir: Virtual skill directory to scan.

    Returns:
        A compiled LangGraph agent.
    """
    fs_tools = _guard_writes(build_fs_tools(backend))
    skills_index, read_skill = build_skills(backend, skill_dir=skill_dir)
    full_prompt = f"{system_prompt}\n\nAvailable skills:\n{skills_index}"

    return create_agent(
        model=model,
        tools=[*fs_tools, read_skill, write_todos],
        system_prompt=full_prompt,
        middleware=list(middleware),
        state_schema=AnalyticsState,
        name=name,
    )


def build_orchestrator(
    *,
    system_prompt: str,
    model: BaseChatModel,
    middleware: Sequence[AgentMiddleware],
    backend: BackendProtocol,
    subagents: Mapping[str, CompiledStateGraph],
    skill_dir: str,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph:
    """Build the orchestrator graph with the task tool.

    Args:
        system_prompt: Orchestrator instructions.
        model: Chat model.
        middleware: Retry/fallback/lifecycle middleware.
        backend: Shared filesystem/sandbox backend.
        subagents: Name-to-graph registry used by task.
        skill_dir: Virtual orchestrator skill directory.
        checkpointer: Optional LangGraph checkpointer.

    Returns:
        A compiled LangGraph agent.
    """
    fs_tools = _guard_writes(build_fs_tools(backend))
    skills_index, read_skill = build_skills(backend, skill_dir=skill_dir)
    task = build_task_tool(subagents)
    available_subagents = "\n".join(f"- {name}" for name in subagents)
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Available subagents (call via task):\n{available_subagents}\n\n"
        f"Available skills:\n{skills_index}"
    )

    return create_agent(
        model=model,
        tools=[*fs_tools, read_skill, write_todos, task],
        system_prompt=full_prompt,
        middleware=list(middleware),
        state_schema=AnalyticsState,
        checkpointer=checkpointer,
        name="orchestrator",
    )
