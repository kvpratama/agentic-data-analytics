"""Pure-LangGraph EDA orchestrator, parallel to ada.agent."""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import Awaitable, Callable
from typing import cast

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
)
from langchain_core.runnables import RunnableConfig
from langchain_modal import ModalSandbox
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from ada.agent_middleware import SandboxLifecycleMiddleware
from ada.config import get_model, get_model_small, get_settings
from ada.runtime.workspace import provision_workspace
from ada_lg.lg.agent_factory import build_orchestrator, build_subagent
from ada_lg.subagents import get_subagent_specs

_SCHEMA_GRAPH_CACHE: CompiledStateGraph | None = None

_ORCHESTRATOR_PROMPT = """\
You are the Data Analytics Orchestrator. You have an `execute` tool (runs shell
commands inside an isolated sandbox) and three subagents: profiler, cleaner,
and analyst. All dataset files live under '/workspace/'

Load the 'orchestrator' skill before deciding how to proceed. It contains your
decision framework, routing guidance, and examples.

Your goal is to satisfy the user's objective — which may be a specific question,
an instruction, or a full EDA request — using whichever combination of tools and
subagents is appropriate. The final deliverable is either a direct answer, a
report.md, or both."""


async def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Build a fresh per-turn analytics graph for LangGraph Studio.

    Args:
        config: Runnable config with execution metadata under configurable.

    Returns:
        A compiled analytics graph.
    """
    configurable = dict(config.get("configurable", {}))
    is_execution = bool(configurable.get("__is_for_execution__", False))
    if is_execution:
        if "thread_id" not in configurable:
            msg = "make_graph execution requires configurable.thread_id"
            raise ValueError(msg)
        thread_id = str(configurable["thread_id"])

        if "csv_path" not in configurable:
            msg = "make_graph execution requires configurable.csv_path"
            raise ValueError(msg)
        csv_path = await asyncio.to_thread(
            lambda: pathlib.Path(str(configurable["csv_path"])).resolve()
        )

        if "stem" not in configurable:
            configurable["stem"] = csv_path.stem
        stem = str(configurable["stem"])

        sandbox_resources, mirror_root = await provision_workspace(stem, thread_id, csv_path)
        return await asyncio.to_thread(
            create_analytics_agent,
            sandbox_resources.backend,
            mirror_root=mirror_root,
            terminate_sandbox=sandbox_resources.terminate,
        )

    global _SCHEMA_GRAPH_CACHE
    if _SCHEMA_GRAPH_CACHE is None:
        _SCHEMA_GRAPH_CACHE = await asyncio.to_thread(
            create_analytics_agent,
            StateBackend(),
            mirror_root=None,
        )
    return _SCHEMA_GRAPH_CACHE


def create_analytics_agent(
    backend: BackendProtocol,
    *,
    mirror_root: pathlib.Path | None = None,
    terminate_sandbox: Callable[[], Awaitable[None]] | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph:
    """Build the pure-LangGraph analytics orchestrator.

    Args:
        backend: Shared sandbox/filesystem backend.
        mirror_root: Optional host-side artifact mirror directory.
        terminate_sandbox: Async sandbox terminator required with mirror_root.
        checkpointer: Optional LangGraph checkpointer.

    Returns:
        A compiled orchestrator graph.
    """
    if mirror_root is not None and terminate_sandbox is None:
        msg = "create_analytics_agent requires terminate_sandbox with mirror_root"
        raise ValueError(msg)

    settings = get_settings()
    model = get_model(settings)
    model_small = get_model_small(settings)

    composite_backend = CompositeBackend(
        default=backend,
        routes={
            "/skills/": FilesystemBackend(
                root_dir=str(pathlib.Path(__file__).resolve().parent.parent / "skills"),
                virtual_mode=True,
            ),
        },
    )

    base_middleware: list[AgentMiddleware] = [
        ModelRetryMiddleware(
            max_retries=settings.retry_max_retries,
            backoff_factor=settings.retry_backoff_factor,
            initial_delay=settings.retry_initial_delay,
        ),
        ModelFallbackMiddleware(model_small),
    ]

    orchestrator_middleware = list(base_middleware)
    if mirror_root is not None:
        assert terminate_sandbox is not None  # noqa: S101 — validated above
        orchestrator_middleware.append(
            SandboxLifecycleMiddleware(
                backend=cast("ModalSandbox", backend),
                mirror_root=mirror_root,
                terminate=terminate_sandbox,
            )
        )

    subagent_graphs: dict[str, CompiledStateGraph] = {}
    for spec in get_subagent_specs():
        subagent_graphs[spec["name"]] = build_subagent(
            name=spec["name"],
            system_prompt=spec["system_prompt"],
            model=model,
            middleware=base_middleware,
            backend=composite_backend,
            skill_dir=spec["skill_dir"],
        )

    return build_orchestrator(
        system_prompt=_ORCHESTRATOR_PROMPT,
        model=model,
        middleware=orchestrator_middleware,
        backend=composite_backend,
        subagents=subagent_graphs,
        skill_dir="/skills/orchestrator_skills/",
        checkpointer=checkpointer,
    )
