"""Multi-subagent EDA orchestrator using Deep Agents and a Modal sandbox.

Three subagents — profiler, cleaner, and analyst — share a single ephemeral
Modal microVM (``ModalSandbox`` backend). Each subagent loads its own
SKILL.md (progressive disclosure) for methodology and pandas/scipy snippets.

Usage:
    python cli.py <csv_path> <objective>

Example:
    python cli.py dataset/Titanic-Dataset.csv "Investigate factors that affected survival"
"""

import asyncio
import pathlib
import tempfile
from collections.abc import Awaitable, Callable
from typing import cast

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import BackendProtocol, CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
)
from langchain_core.runnables import RunnableConfig
from langchain_modal import ModalSandbox
from langgraph.graph.state import CompiledStateGraph

from agent_middleware import SandboxLifecycleMiddleware
from config import get_model, get_model_small, get_settings
from runtime.workspace import provision_workspace
from subagents import get_subagents

# Cached schema-only graph reused for every Studio read call
# (assistants.read, threads.read, threads.update). The topology cannot change
# at runtime; on code edits langgraph dev hot-reloads the module and resets it.
_SCHEMA_GRAPH_CACHE: CompiledStateGraph | None = None

# Prime tempfile.tempdir at import time so later async code (e.g. Modal's
# Resolver creating a TemporaryFile) does not hit os.getcwd() on the event
# loop, which blockbuster flags as a blocking call under `langgraph dev`.
tempfile.gettempdir()


async def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Build a fresh per-turn analytics graph for LangGraph Studio.

    Args:
        config: Runnable config with ``configurable.csv_path``,
            ``configurable.stem``, and ``configurable.thread_id``.

    Returns:
        A compiled analytics graph backed by a freshly seeded Modal sandbox.
    """
    configurable = dict(config.get("configurable", {}))
    # Default to False (fail-closed): only provision a real Modal sandbox when
    # the caller explicitly signals this is an execution. Schema/read calls
    # from LangGraph Studio (assistants.read, threads.read, threads.update)
    # fall through to the cached no-op SchemaOnlySandboxBackend graph.
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
        return create_analytics_agent(
            sandbox_resources.backend,
            mirror_root=mirror_root,
            terminate_sandbox=sandbox_resources.terminate,
        )

    global _SCHEMA_GRAPH_CACHE
    if _SCHEMA_GRAPH_CACHE is None:
        _SCHEMA_GRAPH_CACHE = create_analytics_agent(StateBackend(), mirror_root=None)
    return _SCHEMA_GRAPH_CACHE


def create_analytics_agent(
    backend: BackendProtocol,
    *,
    mirror_root: pathlib.Path | None = None,
    terminate_sandbox: Callable[[], Awaitable[None]] | None = None,
) -> CompiledStateGraph:
    """Build the Deep Agent orchestrator with profiler, cleaner, and analyst subagents.

    Args:
        backend: A live ``ModalSandbox`` backend shared across all subagents.
            All filesystem operations and the auto-injected ``execute`` tool
            route through this sandbox.
        mirror_root: Optional host-side mirror directory. When supplied, a
            lifecycle middleware downloads artifacts there and terminates the
            sandbox after the turn.
        terminate_sandbox: Async callable that releases the real Modal sandbox.

    Returns:
        A configured Deep Agent ready to invoke with a user objective.
    """
    settings = get_settings()
    model = get_model(settings)
    model_small = get_model_small(settings)

    subagents = get_subagents(settings)

    middleware: list[AgentMiddleware] = [
        ModelRetryMiddleware(
            max_retries=settings.retry_max_retries,
            backoff_factor=settings.retry_backoff_factor,
            initial_delay=settings.retry_initial_delay,
        ),
        ModelFallbackMiddleware(model_small),
    ]
    if mirror_root is not None:
        if terminate_sandbox is None:
            msg = "create_analytics_agent requires terminate_sandbox with mirror_root"
            raise ValueError(msg)
        modal_backend = cast("ModalSandbox", backend)
        middleware.append(
            SandboxLifecycleMiddleware(
                backend=modal_backend,
                mirror_root=mirror_root,
                terminate=terminate_sandbox,
            )
        )

    return create_deep_agent(
        model=model,
        system_prompt="""\
You are the Data Analytics Orchestrator. You have an `execute` tool (runs shell
commands inside an isolated sandbox) and three subagents: profiler, cleaner,
and analyst.

Load the 'orchestrator' skill before deciding how to proceed. It contains your
decision framework, routing guidance, and examples.

Your goal is to satisfy the user's objective — which may be a specific question,
an instruction, or a full EDA request — using whichever combination of tools and
subagents is appropriate. The final deliverable is either a direct answer, a
report.md, or both.""",
        middleware=middleware,
        subagents=subagents,
        backend=CompositeBackend(
            default=backend,
            routes={
                "/skills/": FilesystemBackend(
                    root_dir=str(pathlib.Path(__file__).resolve().parent / "skills"),
                    virtual_mode=True,
                ),
            },
        ),
        skills=["/skills/orchestrator_skills/"],
        permissions=[
            FilesystemPermission(
                operations=["write"],
                paths=["/skills/**"],
                mode="deny",
            ),
        ],
    )
