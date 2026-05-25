"""Tests for the analytics orchestrator wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_modal import ModalSandbox

from agent import create_analytics_agent
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
    """The supplied ModalSandbox backend is forwarded to create_deep_agent."""
    backend = MagicMock(spec=ModalSandbox)
    mock_create, captured = _capture_create_deep_agent()

    with (
        patch("agent.get_settings", return_value=Settings(_env_file=None)),  # type: ignore
        patch("agent.get_model", return_value=MagicMock()),
        patch("agent.get_model_small", return_value=MagicMock()),
        patch("agent.create_deep_agent", mock_create),
    ):
        create_analytics_agent(backend)

    assert captured["backend"] is backend
