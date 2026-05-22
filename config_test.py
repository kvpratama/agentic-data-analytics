"""Tests for the Settings object."""

import pytest

from config import Settings


def test_default_modal_settings() -> None:
    """modal_app_name and modal_sandbox_timeout have sensible defaults."""
    s = Settings(_env_file=None)  # type: ignore
    assert s.modal_app_name == "agentic-data-analytics"
    assert s.modal_sandbox_timeout == 60 * 30


def test_modal_settings_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both settings can be overridden via environment variables."""
    monkeypatch.setenv("MODAL_APP_NAME", "custom-app")
    monkeypatch.setenv("MODAL_SANDBOX_TIMEOUT", "120")
    s = Settings(_env_file=None)  # type: ignore
    assert s.modal_app_name == "custom-app"
    assert s.modal_sandbox_timeout == 120
