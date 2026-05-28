"""Tests for analytics agent lifecycle middleware."""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_middleware import SandboxLifecycleMiddleware


def _backend_double() -> MagicMock:
    """Create a backend double matching ``langchain_modal.ModalSandbox`` internals."""
    backend = MagicMock()
    backend._sandbox.terminate.aio = AsyncMock()
    return backend


async def test_sandbox_lifecycle_middleware_downloads_then_terminates(
    tmp_path: pathlib.Path,
) -> None:
    """The after-agent hook mirrors files before releasing the sandbox."""
    backend = _backend_double()
    downloader = AsyncMock()
    middleware = SandboxLifecycleMiddleware(
        backend=backend,
        mirror_root=tmp_path,
        downloader=downloader,
    )

    await middleware.aafter_agent({}, MagicMock())

    downloader.assert_awaited_once_with(backend, local_root=tmp_path)
    backend._sandbox.terminate.aio.assert_awaited_once_with()


async def test_sandbox_lifecycle_middleware_terminates_when_download_fails(
    tmp_path: pathlib.Path,
) -> None:
    """Sandbox teardown still happens if artifact mirroring raises."""
    backend = _backend_double()
    downloader = AsyncMock(side_effect=RuntimeError("download failed"))
    middleware = SandboxLifecycleMiddleware(
        backend=backend,
        mirror_root=tmp_path,
        downloader=downloader,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        await middleware.aafter_agent({}, MagicMock())

    backend._sandbox.terminate.aio.assert_awaited_once_with()
