import asyncio
import contextlib
import pathlib
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import modal
from deepagents.backends import BackendProtocol
from langchain_modal import ModalSandbox

from ada.config import get_settings
from ada.runtime.backend import has_modal_credentials
from ada.runtime.local_runtime import LocalWorkspaceBackend
from ada.runtime.modal_runtime import build_image, seed_sandbox


@dataclass(frozen=True)
class SandboxResources:
    """Sandbox backend plus the explicit sandbox teardown callable."""

    backend: BackendProtocol
    terminate: Callable[[], Awaitable[None]]


def _project_root() -> pathlib.Path:
    """Return the repository root containing this module."""
    return pathlib.Path(__file__).resolve().parent.parent.parent


_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_path_component(value: str, *, field: str) -> str:
    """Validate a single path component against the safe-character allowlist.

    Args:
        value: Candidate path component.
        field: Field name used in the error message.

    Returns:
        The validated value unchanged.

    Raises:
        ValueError: If ``value`` contains characters outside ``[A-Za-z0-9._-]``.
    """
    if not _SAFE_PATH_COMPONENT.fullmatch(value):
        raise ValueError(f"invalid {field}: {value!r}")
    return value


def get_mirror_root(stem: str, thread_id: str) -> pathlib.Path:
    """Return the host-side per-thread workspace directory.

    Args:
        stem: Dataset filename stem (e.g. ``"Titanic-Dataset"``).
        thread_id: Unique LangGraph thread identifier.

    Returns:
        Absolute path under ``<project>/workspace/<stem>_<thread_id>``.
    """
    safe_stem = _validate_path_component(stem, field="stem")
    safe_thread_id = _validate_path_component(thread_id, field="thread_id")
    return _project_root() / "workspace" / f"{safe_stem}_{safe_thread_id}"


def bootstrap_mirror(mirror_root: pathlib.Path, csv_path: pathlib.Path) -> None:
    """Create a thread mirror and copy the raw CSV into it on first use.

    Args:
        mirror_root: Host-side per-thread workspace directory.
        csv_path: Source CSV path supplied by the caller.
    """
    mirror_root.mkdir(parents=True, exist_ok=True)
    dataset = mirror_root / "dataset.csv"
    if not dataset.exists():
        shutil.copyfile(csv_path, dataset)


async def create_sandbox(thread_id: str) -> SandboxResources:
    """Create a fresh Modal sandbox backend for one graph turn.

    Args:
        thread_id: LangGraph thread ID used as a Modal sandbox tag.

    Returns:
        A ``ModalSandbox`` backend and explicit sandbox teardown callable.
    """
    settings = get_settings()
    app = await modal.App.lookup.aio(settings.modal_app_name, create_if_missing=True)
    modal_sandbox = await modal.Sandbox.create.aio(
        image=build_image(),
        app=app,
        tags={"thread_id": thread_id},
        timeout=settings.modal_sandbox_timeout,
        cpu=2.0,
        memory=4096,
    )
    return SandboxResources(
        backend=ModalSandbox(sandbox=modal_sandbox),
        terminate=modal_sandbox.terminate.aio,
    )


def _setup_local_workspace(ws_dir: pathlib.Path, csv_path: pathlib.Path) -> None:
    """Set up the local workspace directory and copy the CSV in a synchronous thread.

    Args:
        ws_dir: Absolute path to the /workspace directory under the mirror root.
        csv_path: Source CSV path.
    """
    ws_dir.mkdir(parents=True, exist_ok=True)
    dataset_dest = ws_dir / "dataset.csv"
    if not dataset_dest.exists():
        shutil.copyfile(csv_path, dataset_dest)


async def _provision_local(
    stem: str,
    thread_id: str,
    csv_path: pathlib.Path,
    mirror_root: pathlib.Path,
) -> SandboxResources:
    """Provision a LocalShellBackend workspace instead of a Modal sandbox.

    Args:
        stem: Dataset filename stem.
        thread_id: LangGraph thread ID.
        csv_path: Source CSV path.
        mirror_root: Host-side per-thread workspace directory.

    Returns:
        A ``SandboxResources`` with a ``LocalWorkspaceBackend`` and no-op terminate.
    """
    # Create /workspace/ subdirectory inside mirror_root so virtual_mode maps correctly
    ws_dir = mirror_root / "workspace"
    await asyncio.to_thread(_setup_local_workspace, ws_dir, csv_path)
    backend = LocalWorkspaceBackend(workspace_path=mirror_root)
    return SandboxResources(backend=backend, terminate=lambda: _async_noop())


async def _async_noop() -> None:
    """No-op async callable for local backends that have no sandbox lifecycle."""
    return


async def provision_workspace(
    stem: str, thread_id: str, csv_path: pathlib.Path
) -> tuple[SandboxResources, pathlib.Path]:
    """Provision the host mirror and seed either a Modal or local backend.

    Args:
        stem: Dataset filename stem (e.g. ``"Titanic-Dataset"``).
        thread_id: Unique LangGraph thread identifier.
        csv_path: Source CSV path supplied by the caller.

    Returns:
        A tuple of (SandboxResources, mirror_root path).
    """
    mirror_root = get_mirror_root(stem, thread_id)
    await asyncio.to_thread(bootstrap_mirror, mirror_root, csv_path)

    if has_modal_credentials():
        sandbox_resources = await create_sandbox(thread_id)
        try:
            await seed_sandbox(
                cast(ModalSandbox, sandbox_resources.backend), mirror_root=mirror_root
            )
        except Exception:
            with contextlib.suppress(Exception):
                await sandbox_resources.terminate()
            raise
        return sandbox_resources, mirror_root
    else:
        resources = await _provision_local(stem, thread_id, csv_path, mirror_root)
        return resources, mirror_root
