"""Tests for ada_lg.agent.create_analytics_agent and make_graph."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_modal import ModalSandbox
from langgraph.checkpoint.base import BaseCheckpointSaver

from ada.config import Settings
from ada_lg.agent import create_analytics_agent, make_graph


async def _run_to_thread_sync[**P, T](
    func: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run a to_thread target inline for unit tests."""
    return func(*args, **kwargs)


def test_create_analytics_agent_builds_subagents_and_orchestrator() -> None:
    """create_analytics_agent wires three subagents into the orchestrator."""
    settings = Settings()
    backend = MagicMock(spec=ModalSandbox)
    orchestrator = MagicMock(name="orchestrator")

    with (
        patch("ada_lg.agent.get_settings", return_value=settings),
        patch("ada_lg.agent.get_model", return_value=MagicMock(name="primary")),
        patch("ada_lg.agent.get_model_small", return_value=MagicMock(name="small")),
        patch(
            "ada_lg.agent.build_subagent", side_effect=[MagicMock(), MagicMock(), MagicMock()]
        ) as build_subagent,
        patch("ada_lg.agent.build_orchestrator", return_value=orchestrator) as build_orchestrator,
    ):
        graph = create_analytics_agent(backend, checkpointer=MagicMock(spec=BaseCheckpointSaver))

    assert graph is orchestrator
    assert [call.kwargs["name"] for call in build_subagent.call_args_list] == [
        "profiler",
        "cleaner",
        "analyst",
    ]
    orchestrator_kwargs = build_orchestrator.call_args.kwargs
    assert set(orchestrator_kwargs["subagents"]) == {"profiler", "cleaner", "analyst"}
    assert orchestrator_kwargs["checkpointer"] is not None


def test_create_analytics_agent_wires_retry_fallback_and_composite_backend() -> None:
    """Middleware and backend routing match the reference implementation."""
    settings = Settings(
        retry_max_retries=7,
        retry_backoff_factor=3.0,
        retry_initial_delay=2.5,
    )
    backend = MagicMock(spec=ModalSandbox)

    with (
        patch("ada_lg.agent.get_settings", return_value=settings),
        patch("ada_lg.agent.get_model", return_value=MagicMock(name="primary")),
        patch("ada_lg.agent.get_model_small", return_value=MagicMock(name="small")),
        patch("ada_lg.agent.build_subagent", return_value=MagicMock()),
        patch("ada_lg.agent.build_orchestrator", return_value=MagicMock()) as build_orchestrator,
    ):
        create_analytics_agent(backend)

    kwargs = build_orchestrator.call_args.kwargs
    composite = kwargs["backend"]
    assert isinstance(composite, CompositeBackend)
    assert composite.default is backend
    assert isinstance(composite.routes["/skills/"], FilesystemBackend)

    middleware = kwargs["middleware"]
    retry = next(item for item in middleware if isinstance(item, ModelRetryMiddleware))
    assert retry.max_retries == 7
    assert retry.backoff_factor == 3.0
    assert retry.initial_delay == 2.5
    assert any(isinstance(item, ModelFallbackMiddleware) for item in middleware)


def test_create_analytics_agent_requires_terminator_with_mirror_root(
    tmp_path: pathlib.Path,
) -> None:
    """A mirror root without a termination callback is invalid."""
    with pytest.raises(ValueError, match="terminate_sandbox"):
        create_analytics_agent(MagicMock(spec=ModalSandbox), mirror_root=tmp_path)


async def test_make_graph_for_studio_introspection_does_not_create_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema reads use a cached StateBackend graph."""
    graph = MagicMock(name="CompiledStateGraph")
    monkeypatch.setattr("ada_lg.agent._SCHEMA_GRAPH_CACHE", None)

    with (
        patch("ada_lg.agent.asyncio.to_thread", side_effect=_run_to_thread_sync) as to_thread,
        patch("ada_lg.agent.provision_workspace", new=AsyncMock()) as provision,
        patch("ada_lg.agent.create_analytics_agent", return_value=graph) as create_agent,
    ):
        result = await make_graph({})
        result2 = await make_graph({})

    assert result is graph
    assert result2 is graph
    provision.assert_not_awaited()
    create_agent.assert_called_once()
    assert to_thread.call_args_list[0].args[0] is create_agent
    assert isinstance(create_agent.call_args.args[0], StateBackend)
    assert create_agent.call_args.kwargs == {"mirror_root": None}


async def test_make_graph_execution_requires_csv_path() -> None:
    """Execution configs must include csv_path before provisioning."""
    with patch("ada_lg.agent.provision_workspace", new=AsyncMock()) as provision:
        with pytest.raises(ValueError, match="csv_path"):
            await make_graph(
                {
                    "configurable": {
                        "__is_for_execution__": True,
                        "thread_id": "thread-1",
                    }
                }
            )
    provision.assert_not_awaited()


async def test_make_graph_creates_sandbox_and_graph(tmp_path: pathlib.Path) -> None:
    """Execution path provisions a workspace and builds a real graph."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    backend = MagicMock(spec=ModalSandbox)
    graph = MagicMock(name="CompiledStateGraph")

    from ada.runtime.workspace import SandboxResources

    terminate = AsyncMock()
    resources = SandboxResources(backend=backend, terminate=terminate)
    mirror_root = tmp_path / "workspace" / "input_thread-1"

    with (
        patch("ada_lg.agent.asyncio.to_thread", side_effect=_run_to_thread_sync) as to_thread,
        patch(
            "ada_lg.agent.provision_workspace", new=AsyncMock(return_value=(resources, mirror_root))
        ) as provision,
        patch("ada_lg.agent.create_analytics_agent", return_value=graph) as create_agent,
    ):
        result = await make_graph(
            {
                "configurable": {
                    "__is_for_execution__": True,
                    "csv_path": str(csv),
                    "stem": "input",
                    "thread_id": "thread-1",
                }
            }
        )

    assert result is graph
    assert to_thread.call_args_list[-1].args[0] is create_agent
    provision.assert_awaited_once_with("input", "thread-1", csv.resolve())
    create_agent.assert_called_once_with(
        backend,
        mirror_root=mirror_root,
        terminate_sandbox=terminate,
    )
