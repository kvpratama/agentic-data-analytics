"""Middleware for mirroring sandbox artifacts and releasing Modal sandboxes."""

from __future__ import annotations

import pathlib
from collections.abc import Awaitable, Callable
from typing import Protocol, cast

from langchain.agents.middleware import AgentMiddleware
from langchain_modal import ModalSandbox

from runtime.modal_runtime import download_artifacts


class _AsyncTerminate(Protocol):
    """Protocol for Modal's async terminate handle."""

    async def aio(self) -> None:
        """Terminate the sandbox asynchronously."""


class _SandboxTerminateHandle(Protocol):
    """Protocol for the sandbox object exposed by ``ModalSandbox``."""

    terminate: _AsyncTerminate


class _BackendWithSandboxHandle(Protocol):
    """Protocol for the instance-only ``ModalSandbox._sandbox`` attribute."""

    _sandbox: _SandboxTerminateHandle


class SandboxLifecycleMiddleware(AgentMiddleware):
    """Mirror ``/work/`` artifacts to the host, then terminate the sandbox."""

    def __init__(
        self,
        *,
        backend: ModalSandbox,
        mirror_root: pathlib.Path,
        downloader: Callable[..., Awaitable[list[pathlib.Path]]] = download_artifacts,
    ) -> None:
        """Initialize the middleware.

        Args:
            backend: The live Modal sandbox backend used by the agent turn.
            mirror_root: Host-side thread mirror directory to receive artifacts.
            downloader: Async artifact download function. Injectable for tests.
        """
        super().__init__()
        self.backend = backend
        self.mirror_root = mirror_root
        self.downloader = downloader

    async def aafter_agent(self, state: object, runtime: object) -> dict[str, object] | None:
        """Download artifacts after a turn and always release the sandbox.

        Args:
            state: Current agent state from LangChain.
            runtime: Current agent runtime from LangChain.

        Returns:
            No state updates.
        """
        del state, runtime
        try:
            await self.downloader(self.backend, local_root=self.mirror_root)
        finally:
            backend = cast("_BackendWithSandboxHandle", self.backend)
            await backend._sandbox.terminate.aio()
        return None
