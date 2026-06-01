"""Shared configuration for the agentic data analytics example.

Loads settings from a local ``.env`` file so you can swap models or providers
without editing any source file.

Examples (.env):

    # Default: Anthropic Sonnet 4.5 (uses ANTHROPIC_API_KEY from env)
    MODEL=anthropic:claude-sonnet-4-5-20250929

    # Switch to OpenAI (uses OPENAI_API_KEY from env)
    MODEL=openai:gpt-4o

    # Switch to Google (uses GOOGLE_API_KEY from env)
    MODEL=google_genai:gemini-2.0-flash
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# NOTE: env loading is deliberately NOT performed at import time. Callers
# (cli.main, tests, langgraph dev startup) invoke load_environment() once
# at the right time so the cwd-relative ./.env and user-global config are
# read with explicit ordering.


def load_environment() -> None:
    """Populate ``os.environ`` from the layered config sources.

    Resolution order (highest priority first):

    1. OS environment variables already set in the process.
    2. ``./.env`` in the current working directory (project-local overrides).
    3. ``~/.config/ada/config.env`` (user-global config).

    Both ``load_dotenv`` calls use ``override=False`` so existing keys are
    never replaced. The local ``./.env`` is loaded *before* the user config,
    so it claims the available keys first — making ``./.env`` the second-
    tier source.
    """
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env, override=False)  # ./.env fills gaps the OS didn't set
    user_config = Path.home() / ".config" / "ada" / "config.env"
    if user_config.exists():
        load_dotenv(user_config, override=False)


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        model: LangChain model identifier with provider prefix
            (e.g. ``anthropic:claude-sonnet-4-5-20250929``).
        model_provider: Optional explicit provider override. Usually inferred
            from the ``model`` prefix.
        base_url: Optional API base URL (for proxies or self-hosted endpoints).
        api_key: Optional API key. If unset, ``init_chat_model`` falls back to
            the provider's standard env var.
        temperature: Sampling temperature.
        modal_app_name: Name of the Modal app that owns sandboxes for this
            project. Created on first use if missing.
        modal_sandbox_timeout: Hard wall-clock cap (seconds) on a single
            sandbox's lifetime. Defaults to 30 minutes.
        retry_max_retries: Maximum number of retries for ModelRetryMiddleware.
        retry_backoff_factor: Exponential backoff factor for ModelRetryMiddleware.
        retry_initial_delay: Initial delay (seconds) for ModelRetryMiddleware.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model: str = "anthropic:claude-sonnet-4-5-20250929"
    model_small: str = "anthropic:claude-3-5-sonnet-20241022"
    model_provider: str | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    temperature: float = 0.0
    modal_app_name: str = "agentic-data-analytics"
    modal_sandbox_timeout: int = 60 * 30
    retry_max_retries: int = 5
    retry_backoff_factor: float = 2.0
    retry_initial_delay: float = 5.0


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()


def _build_model(settings: Settings, model_id: str) -> BaseChatModel:
    """Build a LangChain chat model from the current Settings.

    Only forwards optional fields that are explicitly set, so unset values
    fall through to ``init_chat_model``'s default credential resolution.

    Args:
        settings: The Settings object containing configuration.
        model_id: The model identifier to use (e.g., from settings.model or
            settings.model_small).

    Returns:
        A configured BaseChatModel instance.
    """
    kwargs: dict = {"model": model_id, "temperature": settings.temperature}
    if settings.model_provider:
        kwargs["model_provider"] = settings.model_provider
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key is not None:
        kwargs["api_key"] = settings.api_key.get_secret_value()
    return init_chat_model(**kwargs)


def get_model(settings: Settings | None = None) -> BaseChatModel:
    """Build the primary LangChain chat model from the current Settings.

    Args:
        settings: Optional Settings instance. If None, calls get_settings().

    Returns:
        A configured BaseChatModel instance using the primary model
        identifier from settings.
    """
    if settings is None:
        settings = get_settings()
    return _build_model(settings, settings.model)


def get_model_small(settings: Settings | None = None) -> BaseChatModel:
    """Build the small LangChain chat model from the current Settings.

    Args:
        settings: Optional Settings instance. If None, calls get_settings().

    Returns:
        A configured BaseChatModel instance using the small model
        identifier from settings.
    """
    if settings is None:
        settings = get_settings()
    return _build_model(settings, settings.model_small)
