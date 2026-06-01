"""Tests for the Settings object and model builders."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel

from ada.config import Settings, get_model, get_model_small, load_environment


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


def test_default_model_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary and small model identifiers have sensible defaults."""
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("MODEL_SMALL", raising=False)
    s = Settings(_env_file=None)  # type: ignore
    assert s.model == "anthropic:claude-sonnet-4-5-20250929"
    assert s.model_small == "anthropic:claude-3-5-sonnet-20241022"


def test_model_small_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """model_small can be overridden via the MODEL_SMALL env var."""
    monkeypatch.setenv("MODEL_SMALL", "anthropic:claude-3-haiku-20240307")
    s = Settings(_env_file=None)  # type: ignore
    assert s.model_small == "anthropic:claude-3-haiku-20240307"


def test_default_retry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelRetryMiddleware tuning knobs have sensible defaults."""
    monkeypatch.delenv("RETRY_MAX_RETRIES", raising=False)
    monkeypatch.delenv("RETRY_BACKOFF_FACTOR", raising=False)
    monkeypatch.delenv("RETRY_INITIAL_DELAY", raising=False)
    s = Settings(_env_file=None)  # type: ignore
    assert s.retry_max_retries == 5
    assert s.retry_backoff_factor == 2.0
    assert s.retry_initial_delay == 5.0


def test_retry_settings_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """All three retry settings can be overridden via environment variables."""
    monkeypatch.setenv("RETRY_MAX_RETRIES", "10")
    monkeypatch.setenv("RETRY_BACKOFF_FACTOR", "3.5")
    monkeypatch.setenv("RETRY_INITIAL_DELAY", "1.25")
    s = Settings(_env_file=None)  # type: ignore
    assert s.retry_max_retries == 10
    assert s.retry_backoff_factor == 3.5
    assert s.retry_initial_delay == 1.25


def test_get_model_uses_primary_model_id() -> None:
    """get_model() forwards the Settings.model id to init_chat_model."""
    s = Settings(_env_file=None, model="anthropic:test-primary")  # type: ignore
    with patch("ada.config.init_chat_model") as mock_init:
        mock_init.return_value = object()
        get_model(s)
    assert mock_init.call_args.kwargs["model"] == "anthropic:test-primary"


def test_get_model_small_uses_small_model_id() -> None:
    """get_model_small() forwards the Settings.model_small id to init_chat_model."""
    s = Settings(_env_file=None, model_small="anthropic:test-small")  # type: ignore
    with patch("ada.config.init_chat_model") as mock_init:
        mock_init.return_value = object()
        get_model_small(s)
    assert mock_init.call_args.kwargs["model"] == "anthropic:test-small"


def test_get_model_falls_back_to_get_settings_when_none() -> None:
    """When called with no arg, get_model() resolves Settings via get_settings()."""
    fake = Settings(_env_file=None, model="anthropic:from-cache")  # type: ignore
    with (
        patch("ada.config.get_settings", return_value=fake) as mock_get_settings,
        patch("ada.config.init_chat_model") as mock_init,
    ):
        mock_init.return_value = object()
        get_model()
    mock_get_settings.assert_called_once()
    assert mock_init.call_args.kwargs["model"] == "anthropic:from-cache"


def test_get_model_small_falls_back_to_get_settings_when_none() -> None:
    """When called with no arg, get_model_small() resolves Settings via get_settings()."""
    fake = Settings(_env_file=None, model_small="anthropic:small-from-cache")  # type: ignore
    with (
        patch("ada.config.get_settings", return_value=fake) as mock_get_settings,
        patch("ada.config.init_chat_model") as mock_init,
    ):
        mock_init.return_value = object()
        get_model_small()
    mock_get_settings.assert_called_once()
    assert mock_init.call_args.kwargs["model"] == "anthropic:small-from-cache"


def test_get_model_returns_base_chat_model_subclass() -> None:
    """The factory returns the object init_chat_model produced."""
    s = Settings(_env_file=None)  # type: ignore
    sentinel = type("FakeModel", (BaseChatModel,), {})
    with patch("ada.config.init_chat_model", return_value=sentinel):
        assert get_model(s) is sentinel


def test_load_environment_os_env_beats_local_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OS env wins over ./.env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-os")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-local\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "no-home"))

    load_environment()

    assert os.environ["ANTHROPIC_API_KEY"] == "from-os"


def test_load_environment_local_dotenv_beats_user_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """./.env wins over ~/.config/ada/config.env when OS env is unset."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-local\n")
    home = tmp_path / "home"
    user_cfg = home / ".config" / "ada"
    user_cfg.mkdir(parents=True)
    (user_cfg / "config.env").write_text("ANTHROPIC_API_KEY=from-user\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    load_environment()

    assert os.environ["ANTHROPIC_API_KEY"] == "from-local"


def test_load_environment_user_config_fills_when_no_local_or_os(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """~/.config/ada/config.env is consulted when nothing else sets the key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = tmp_path / "home"
    user_cfg = home / ".config" / "ada"
    user_cfg.mkdir(parents=True)
    (user_cfg / "config.env").write_text("OPENAI_API_KEY=from-user\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    load_environment()

    assert os.environ["OPENAI_API_KEY"] == "from-user"


def test_load_environment_tolerates_missing_user_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No user-config file is fine; no error raised."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "no-home"))

    load_environment()  # must not raise
