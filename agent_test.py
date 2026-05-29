"""Tests for the analytics orchestrator wiring."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends import StateBackend
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_modal import ModalSandbox

from agent import create_analytics_agent, make_graph
from config import Settings


async def _run_to_thread_sync[**P, T](
    func: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run a to_thread target inline for unit tests."""
    return func(*args, **kwargs)


def _capture_create_deep_agent() -> tuple[MagicMock, dict[str, Any]]:
    """Return a (mock, last_kwargs) pair for patching create_deep_agent."""
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: object) -> object:
        captured.update(kwargs)
        return MagicMock(name="CompiledStateGraph")

    mock = MagicMock(side_effect=fake_create)
    return mock, captured


def test_create_analytics_agent_wires_retry_and_fallback_middleware_on_orchestrator() -> None:
    """Orchestrator middleware list contains a ModelRetryMiddleware + ModelFallbackMiddleware."""
    settings = Settings(
        _env_file=None,  # type: ignore
        retry_max_retries=7,
        retry_backoff_factor=3.0,
        retry_initial_delay=2.5,
    )
    backend = MagicMock(spec=ModalSandbox)
    mock_create, captured = _capture_create_deep_agent()

    with (
        patch("agent.get_settings", return_value=settings),
        patch("agent.get_model", return_value=MagicMock(name="primary")),
        patch("agent.get_model_small", return_value=MagicMock(name="small")),
        patch("agent.create_deep_agent", mock_create),
    ):
        create_analytics_agent(backend)

    middleware = captured["middleware"]
    retry = next(m for m in middleware if isinstance(m, ModelRetryMiddleware))
    assert retry.max_retries == 7
    assert retry.backoff_factor == 3.0
    assert retry.initial_delay == 2.5
    assert any(isinstance(m, ModelFallbackMiddleware) for m in middleware)


def test_create_analytics_agent_wires_middleware_on_each_subagent() -> None:
    """Every subagent gets its own ModelRetryMiddleware + ModelFallbackMiddleware."""
    settings = Settings(
        _env_file=None,  # type: ignore
        retry_max_retries=4,
        retry_backoff_factor=1.5,
        retry_initial_delay=0.5,
    )
    backend = MagicMock(spec=ModalSandbox)
    mock_create, captured = _capture_create_deep_agent()

    with (
        patch("agent.get_settings", return_value=settings),
        patch("agent.get_model", return_value=MagicMock(name="primary")),
        patch("agent.get_model_small", return_value=MagicMock(name="small")),
        patch("agent.create_deep_agent", mock_create),
    ):
        create_analytics_agent(backend)

    subagents = captured["subagents"]
    names = {sa["name"] for sa in subagents}
    assert names == {"profiler", "cleaner", "analyst"}

    for sa in subagents:
        mw = sa["middleware"]
        retry = next(m for m in mw if isinstance(m, ModelRetryMiddleware))
        assert retry.max_retries == 4
        assert retry.backoff_factor == 1.5
        assert retry.initial_delay == 0.5
        assert any(isinstance(m, ModelFallbackMiddleware) for m in mw), (
            f"{sa['name']} missing ModelFallbackMiddleware"
        )


def test_create_analytics_agent_passes_backend_through() -> None:
    """The supplied ModalSandbox backend is forwarded as the CompositeBackend default,
    with a /skills/ route pointing at the host filesystem."""
    import pathlib

    from deepagents import FilesystemPermission
    from deepagents.backends import CompositeBackend, FilesystemBackend

    backend = MagicMock(spec=ModalSandbox)
    mock_create, captured = _capture_create_deep_agent()

    with (
        patch("agent.get_settings", return_value=Settings(_env_file=None)),  # type: ignore
        patch("agent.get_model", return_value=MagicMock()),
        patch("agent.get_model_small", return_value=MagicMock()),
        patch("agent.create_deep_agent", mock_create),
    ):
        create_analytics_agent(backend)

    composite = captured["backend"]
    assert isinstance(composite, CompositeBackend)
    assert composite.default is backend
    assert "/skills/" in composite.routes
    skills_fs = composite.routes["/skills/"]
    assert isinstance(skills_fs, FilesystemBackend)
    expected_skills_root = pathlib.Path(__file__).resolve().parent / "skills"
    assert pathlib.Path(skills_fs.cwd) == expected_skills_root
    assert skills_fs.virtual_mode is True

    permissions = captured["permissions"]
    assert any(
        isinstance(p, FilesystemPermission)
        and p.mode == "deny"
        and "write" in p.operations
        and any("/skills/" in path for path in p.paths)
        for p in permissions
    ), "Expected a deny-write FilesystemPermission for '/skills/'"


async def test_make_graph_for_studio_introspection_does_not_create_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Studio schema/graph reads can build a graph without provisioning Modal."""
    graph = MagicMock(name="CompiledStateGraph")
    monkeypatch.setattr("agent._SCHEMA_GRAPH_CACHE", None)

    with (
        patch("agent.provision_workspace", new=AsyncMock()) as provision,
        patch("agent.create_analytics_agent", return_value=graph) as create_agent,
    ):
        result = await make_graph({})

    assert result is graph
    provision.assert_not_awaited()
    create_agent.assert_called_once()
    backend = create_agent.call_args.args[0]
    assert isinstance(backend, StateBackend)
    assert create_agent.call_args.kwargs == {"mirror_root": None}


async def test_make_graph_execution_requires_csv_path() -> None:
    """Actual runs fail before provisioning Modal when the dataset config is missing."""
    with patch("agent.provision_workspace", new=AsyncMock()) as provision:
        with pytest.raises(
            ValueError,
            match="make_graph execution requires configurable.csv_path",
        ):
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
    """make_graph calls provision_workspace and passes resources to graph factory."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    backend = MagicMock(spec=ModalSandbox)
    graph = MagicMock(name="CompiledStateGraph")

    from runtime.workspace import SandboxResources

    terminate = AsyncMock()
    resources = SandboxResources(backend=backend, terminate=terminate)
    mirror_root = tmp_path / "workspace" / "input_thread-1"

    with (
        patch("agent.asyncio.to_thread", side_effect=_run_to_thread_sync),
        patch(
            "agent.provision_workspace", new=AsyncMock(return_value=(resources, mirror_root))
        ) as provision,
        patch("agent.create_analytics_agent", return_value=graph) as create_agent,
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
    provision.assert_awaited_once_with("input", "thread-1", csv)
    create_agent.assert_called_once_with(
        backend,
        mirror_root=mirror_root,
        terminate_sandbox=terminate,
    )
