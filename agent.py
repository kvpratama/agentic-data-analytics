"""Multi-subagent EDA orchestrator using Deep Agents and a Modal sandbox.

Three subagents — profiler, cleaner, and analyst — share a single ephemeral
Modal microVM (``ModalSandbox`` backend). Each subagent loads its own
SKILL.md (progressive disclosure) for methodology and pandas/scipy snippets.

Usage:
    python agent.py <csv_path> <objective>

Example:
    python agent.py dataset/Titanic-Dataset.csv "Investigate factors that affected survival"
"""

import asyncio
import contextlib
import os
import pathlib
import shutil
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import modal
from deepagents import FilesystemPermission, SubAgent, create_deep_agent
from deepagents.backends import BackendProtocol, CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
)
from langchain_core.runnables import RunnableConfig
from langchain_modal import ModalSandbox
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.panel import Panel

from agent_middleware import SandboxLifecycleMiddleware
from config import get_model, get_model_small, get_settings
from runtime.modal_runtime import build_image, seed_sandbox

console = Console()

# Cached schema-only graph reused for every Studio read call
# (assistants.read, threads.read, threads.update). The topology cannot change
# at runtime; on code edits langgraph dev hot-reloads the module and resets it.
_SCHEMA_GRAPH_CACHE: CompiledStateGraph | None = None

# Prime tempfile.tempdir at import time so later async code (e.g. Modal's
# Resolver creating a TemporaryFile) does not hit os.getcwd() on the event
# loop, which blockbuster flags as a blocking call under `langgraph dev`.
tempfile.gettempdir()

WORK_RULES = (
    "All dataset files live under '/workspace/'. Use absolute paths: "
    "'/workspace/dataset.csv' (raw, immutable — never overwrite), "
    "'/workspace/dataset.clean.csv' (cleaner output), "
    "'/workspace/profile.json', '/workspace/changes.json', "
    "'/workspace/report.md', '/workspace/plots/'. Skills live under '/skills/'."
)


@dataclass(frozen=True)
class _SandboxResources:
    """Modal backend plus the explicit sandbox teardown callable."""

    backend: ModalSandbox
    terminate: Callable[[], Awaitable[None]]


def _project_root() -> pathlib.Path:
    """Return the repository root containing this module."""
    return pathlib.Path(__file__).resolve().parent


def _mirror_root(stem: str, thread_id: str) -> pathlib.Path:
    """Return the host-side per-thread workspace directory.

    Centralises the naming convention so that ``make_graph`` and ``main``
    always produce identical paths.

    Args:
        stem: Dataset filename stem (e.g. ``"Titanic-Dataset"``).
        thread_id: Unique LangGraph thread identifier.

    Returns:
        Absolute path under ``<project>/workspace/<stem>_<thread_id>``.
    """
    return _project_root() / "workspace" / f"{stem}_{thread_id}"


def _bootstrap_mirror(mirror_root: pathlib.Path, csv_path: pathlib.Path) -> None:
    """Create a thread mirror and copy the raw CSV into it on first use.

    Args:
        mirror_root: Host-side per-thread workspace directory.
        csv_path: Source CSV path supplied by the caller.
    """
    mirror_root.mkdir(parents=True, exist_ok=True)
    dataset = mirror_root / "dataset.csv"
    if not dataset.exists():
        shutil.copyfile(csv_path, dataset)


async def _create_sandbox(thread_id: str) -> _SandboxResources:
    """Create a fresh Modal sandbox backend for one graph turn.

    Args:
        thread_id: LangGraph thread ID used as a Modal sandbox tag.

    Returns:
        A ``ModalSandbox`` backend and explicit sandbox teardown callable.
    """
    settings = get_settings()
    app = await modal.App.lookup.aio(settings.modal_app_name, create_if_missing=True)
    modal_sandbox = await modal.Sandbox.create.aio(
        image=build_image(),
        app=app,
        tags={"thread_id": thread_id},
        timeout=settings.modal_sandbox_timeout,
        cpu=2.0,
        memory=4096,
    )
    return _SandboxResources(
        backend=ModalSandbox(sandbox=modal_sandbox),
        terminate=modal_sandbox.terminate.aio,
    )


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

        mirror_root = _mirror_root(stem, thread_id)
        await asyncio.to_thread(_bootstrap_mirror, mirror_root, csv_path)

        sandbox_resources = await _create_sandbox(thread_id)
        try:
            await seed_sandbox(sandbox_resources.backend, mirror_root=mirror_root)
        except Exception:
            with contextlib.suppress(Exception):
                await sandbox_resources.terminate()
            raise
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

    profiler: SubAgent = {
        "name": "profiler",
        "description": "Profiling agent",
        "system_prompt": (
            "You are a data profiler. Your sole job is to inspect and describe "
            "the dataset as-is — do not clean, transform, or analyse it. "
            "Load the 'profiler' skill for the full methodology and judgement "
            "guidelines, then inspect '/workspace/dataset.csv' and write "
            "'/workspace/profile.json'.\n\n" + WORK_RULES
        ),
        "model": model,
        "middleware": [
            ModelRetryMiddleware(
                max_retries=settings.retry_max_retries,
                backoff_factor=settings.retry_backoff_factor,
                initial_delay=settings.retry_initial_delay,
            ),
            ModelFallbackMiddleware(model_small),
        ],
        "skills": ["/skills/profiler_skills/"],
    }

    cleaner: SubAgent = {
        "name": "cleaner",
        "description": "Cleaning agent",
        "system_prompt": (
            "You are a data cleaner. Your sole job is to fix data quality issues "
            "identified in '/workspace/profile.json' — do not analyse, summarise, or "
            "draw conclusions about the data. Load the 'cleaner' skill for the "
            "full methodology and judgement guidelines. Read '/workspace/profile.json' "
            "first, then apply fixes by reading '/workspace/dataset.csv' (raw, never "
            "modify it) and writing the cleaned output to '/workspace/dataset.clean.csv', "
            "plus '/workspace/changes.json' logging every decision made.\n\n" + WORK_RULES
        ),
        "model": model,
        "middleware": [
            ModelRetryMiddleware(
                max_retries=settings.retry_max_retries,
                backoff_factor=settings.retry_backoff_factor,
                initial_delay=settings.retry_initial_delay,
            ),
            ModelFallbackMiddleware(model_small),
        ],
        "skills": ["/skills/cleaner_skills/"],
    }

    analyst: SubAgent = {
        "name": "analyst",
        "description": "Analyst agent",
        "system_prompt": (
            "You are a data analyst. Your sole job is to analyse and interpret "
            "the cleaned data — do not re-clean or re-profile it. Load the "
            "'analyst' skill for the full methodology and report structure. "
            "Read '/workspace/dataset.clean.csv' and '/workspace/changes.json' (you may "
            "consult '/workspace/dataset.csv' for raw comparisons), then produce "
            "'/workspace/report.md' and save any plots to '/workspace/plots/'. If the "
            "orchestrator passed a specific user question, lead the report with "
            "a direct answer to it.\n\n" + WORK_RULES
        ),
        "model": model,
        "middleware": [
            ModelRetryMiddleware(
                max_retries=settings.retry_max_retries,
                backoff_factor=settings.retry_backoff_factor,
                initial_delay=settings.retry_initial_delay,
            ),
            ModelFallbackMiddleware(model_small),
        ],
        "skills": ["/skills/analyst_skills/"],
    }

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
        subagents=[profiler, cleaner, analyst],
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


async def main() -> None:
    """CLI entrypoint: parse args, build a graph, and stream one turn."""
    if len(sys.argv) < 3:
        console.print("[red]Usage: python agent.py <csv_path> <objective>[/red]")
        sys.exit(1)

    csv_path = sys.argv[1]
    objective = " ".join(sys.argv[2:])

    if not os.path.exists(csv_path):
        console.print(f"[red]Error: CSV file '{csv_path}' not found.[/red]")
        sys.exit(1)

    csv_abs = pathlib.Path(csv_path).resolve()
    stem = csv_abs.stem
    thread_id = str(uuid.uuid4())
    local_root = _mirror_root(stem, thread_id)

    console.print(
        Panel(
            f"[bold blue]Starting EDA for: {stem}[/bold blue]\n"
            f"[italic]Objective: {objective}[/italic]"
        )
    )

    config: RunnableConfig = {
        "configurable": {
            "csv_path": str(csv_abs),
            "stem": stem,
            "thread_id": thread_id,
            "__is_for_execution__": True,
        }
    }
    agent = await make_graph(config)
    async for chunk in agent.astream({"messages": [("user", objective)]}, config=config):
        if "model" in chunk:
            msg = chunk["model"]["messages"][-1]
            if msg.content:
                console.print(f"[dim]{msg.name or 'agent'}:[/dim] {msg.content}")
        elif "tools" in chunk:
            msg = chunk["tools"]["messages"][-1]
            if msg.content:
                console.print(f"[italic]{msg.name or 'agent'}:[/italic] {msg.content}")

    report_path = local_root / "report.md"
    if report_path.exists():
        console.print(
            Panel(
                f"[bold green]Analysis complete![/bold green]\n"
                f"Final report saved to: [underline]{report_path}[/underline]"
            )
        )
    else:
        console.print("[yellow]Warning: report.md was not generated.[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
