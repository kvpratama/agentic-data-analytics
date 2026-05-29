import asyncio
import contextlib
import pathlib
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import modal
from langchain_modal import ModalSandbox

from config import get_settings
from runtime.modal_runtime import build_image, seed_sandbox


@dataclass(frozen=True)
class SandboxResources:
    """Modal backend plus the explicit sandbox teardown callable."""

    backend: ModalSandbox
    terminate: Callable[[], Awaitable[None]]


def _project_root() -> pathlib.Path:
    """Return the repository root containing this module."""
    return pathlib.Path(__file__).resolve().parent.parent


_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_path_component(value: str, *, field: str) -> str:
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


async def provision_workspace(
    stem: str, thread_id: str, csv_path: pathlib.Path
) -> tuple[SandboxResources, pathlib.Path]:
    """Provision the host mirror and seed a fresh Modal sandbox.

    Args:
        stem: Dataset filename stem (e.g. ``"Titanic-Dataset"``).
        thread_id: Unique LangGraph thread identifier.
        csv_path: Source CSV path supplied by the caller.

    Returns:
        A tuple of (SandboxResources, mirror_root path).
    """
    mirror_root = get_mirror_root(stem, thread_id)
    await asyncio.to_thread(bootstrap_mirror, mirror_root, csv_path)

    sandbox_resources = await create_sandbox(thread_id)
    try:
        await seed_sandbox(sandbox_resources.backend, mirror_root=mirror_root)
    except Exception:
        with contextlib.suppress(Exception):
            await sandbox_resources.terminate()
        raise

    return sandbox_resources, mirror_root
