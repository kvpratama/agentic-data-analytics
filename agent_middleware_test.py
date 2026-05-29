"""Tests for analytics agent lifecycle middleware."""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_middleware import SandboxLifecycleMiddleware


def _backend_double() -> MagicMock:
    """Create a backend double matching ``langchain_modal.ModalSandbox``."""
    backend = MagicMock()
    return backend


async def test_sandbox_lifecycle_middleware_downloads_then_terminates(
    tmp_path: pathlib.Path,
) -> None:
    """The after-agent hook mirrors files before releasing the sandbox."""
    backend = _backend_double()
    downloader = AsyncMock()
    terminate = AsyncMock()
    middleware = SandboxLifecycleMiddleware(
        backend=backend,
        mirror_root=tmp_path,
        downloader=downloader,
        terminate=terminate,
    )

    await middleware.aafter_agent({}, MagicMock())

    downloader.assert_awaited_once_with(backend, local_root=tmp_path)
    terminate.assert_awaited_once_with()


async def test_sandbox_lifecycle_middleware_terminates_when_download_fails(
    tmp_path: pathlib.Path,
) -> None:
    """Sandbox teardown still happens if artifact mirroring raises."""
    backend = _backend_double()
    downloader = AsyncMock(side_effect=RuntimeError("download failed"))
    terminate = AsyncMock()
    middleware = SandboxLifecycleMiddleware(
        backend=backend,
        mirror_root=tmp_path,
        downloader=downloader,
        terminate=terminate,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        await middleware.aafter_agent({}, MagicMock())

    terminate.assert_awaited_once_with()
