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

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env so provider SDKs can find ANTHROPIC_API_KEY,
# OPENAI_API_KEY, etc. via their default lookups.
load_dotenv()


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
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model: str = "anthropic:claude-sonnet-4-5-20250929"
    model_provider: str | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    temperature: float = 0.0
    modal_app_name: str = "agentic-data-analytics"
    modal_sandbox_timeout: int = 60 * 30


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()


def get_model() -> BaseChatModel:
    """Build a LangChain chat model from the current Settings.

    Only forwards optional fields that are explicitly set, so unset values
    fall through to ``init_chat_model``'s default credential resolution.
    """
    settings = get_settings()
    kwargs: dict = {"model": settings.model, "temperature": settings.temperature}
    if settings.model_provider:
        kwargs["model_provider"] = settings.model_provider
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key is not None:
        kwargs["api_key"] = settings.api_key.get_secret_value()
    return init_chat_model(**kwargs)
