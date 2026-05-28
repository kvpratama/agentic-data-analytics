"""Tests for the analytics orchestrator wiring."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_modal import ModalSandbox

from agent import _bootstrap_mirror, create_analytics_agent, make_graph
from config import Settings


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


def test_bootstrap_mirror_copies_dataset_once(tmp_path: pathlib.Path) -> None:
    """Mirror bootstrap creates dataset.csv and does not overwrite it later."""
    source = tmp_path / "source.csv"
    source.write_bytes(b"a\n1\n")
    mirror = tmp_path / "work" / "source_thread"

    _bootstrap_mirror(mirror, source)
    assert (mirror / "dataset.csv").read_bytes() == b"a\n1\n"

    source.write_bytes(b"a\n2\n")
    _bootstrap_mirror(mirror, source)
    assert (mirror / "dataset.csv").read_bytes() == b"a\n1\n"


async def test_make_graph_creates_fresh_sandbox_and_seeds_first_turn(
    tmp_path: pathlib.Path,
) -> None:
    """make_graph bootstraps a thread mirror and seeds a fresh Modal sandbox."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    sandbox = MagicMock(name="Sandbox")
    backend = MagicMock(spec=ModalSandbox)
    graph = MagicMock(name="CompiledStateGraph")

    with (
        patch("agent._project_root", return_value=tmp_path),
        patch("agent.modal.App.lookup.aio", new=AsyncMock(return_value=MagicMock())),
        patch("agent.modal.Sandbox.create.aio", new=AsyncMock(return_value=sandbox)) as create,
        patch("agent.ModalSandbox", return_value=backend),
        patch("agent.seed_sandbox", new=AsyncMock()) as seed,
        patch("agent.create_analytics_agent", return_value=graph) as create_agent,
        patch("agent.get_settings", return_value=Settings(_env_file=None)),  # type: ignore
    ):
        result = await make_graph(
            {
                "configurable": {
                    "csv_path": str(csv),
                    "stem": "input",
                    "thread_id": "thread-1",
                }
            }
        )

    mirror = tmp_path / "work" / "input_thread-1"
    assert result is graph
    assert (mirror / "dataset.csv").read_bytes() == b"a,b\n1,2\n"
    create.assert_awaited_once()
    assert create.await_args is not None
    assert create.await_args.kwargs["tags"] == {"thread_id": "thread-1"}
    seed.assert_awaited_once_with(backend, mirror_root=mirror)
    create_agent.assert_called_once_with(backend, mirror_root=mirror)


async def test_make_graph_reuploads_existing_mirror_on_followup(
    tmp_path: pathlib.Path,
) -> None:
    """make_graph keeps thread workspace continuity by seeding accumulated files."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"raw")
    mirror = tmp_path / "work" / "input_thread-1"
    plots = mirror / "plots"
    plots.mkdir(parents=True)
    (mirror / "dataset.csv").write_bytes(b"raw")
    (mirror / "profile.json").write_bytes(b"{}")
    (mirror / "dataset.clean.csv").write_bytes(b"clean")
    (plots / "chart.png").write_bytes(b"png")

    backend = MagicMock(spec=ModalSandbox)

    with (
        patch("agent._project_root", return_value=tmp_path),
        patch("agent.modal.App.lookup.aio", new=AsyncMock(return_value=MagicMock())),
        patch("agent.modal.Sandbox.create.aio", new=AsyncMock(return_value=MagicMock())) as create,
        patch("agent.ModalSandbox", return_value=backend),
        patch("agent.seed_sandbox", new=AsyncMock()) as seed,
        patch("agent.create_analytics_agent", return_value=MagicMock()),
        patch("agent.get_settings", return_value=Settings(_env_file=None)),  # type: ignore
    ):
        await make_graph(
            {
                "configurable": {
                    "csv_path": str(csv),
                    "stem": "input",
                    "thread_id": "thread-1",
                }
            }
        )

    create.assert_awaited_once()
    seed.assert_awaited_once_with(backend, mirror_root=mirror)
