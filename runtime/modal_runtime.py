"""Host-side helpers for driving a Modal sandbox.

Three responsibilities:

* :func:`build_image` declares the Modal Image with the data-science stack
  baked in.
* :func:`seed_sandbox` uploads the dataset into the sandbox.
* :func:`download_artifacts` mirrors the sandbox's ``/work/`` outputs back
  onto the host so users can open ``report.md`` and the plots locally.

All Modal-facing calls go through the ``ModalSandbox`` backend instance the
caller passes in — these helpers do not own the sandbox's lifecycle.

Note: Skills are served to the orchestrator from the host filesystem via a
``CompositeBackend`` route in :mod:`agent`, so they are not uploaded here.
"""

from __future__ import annotations

import asyncio
import pathlib
import shutil

import modal
from langchain_modal import ModalSandbox


def build_image() -> modal.Image:
    """Return the Modal Image used for every sandbox in this project.

    Pins the data-science packages the subagents need. Modal caches the built
    image after the first run, so subsequent sandbox creations are fast.

    Returns:
        A configured ``modal.Image`` ready to pass to ``modal.Sandbox.create``.
    """
    return modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "pandas>=3.0",
        "scipy>=1.17",
        "matplotlib>=3.10",
        "seaborn>=0.13",
        "scikit-learn>=1.8",
    )


async def seed_sandbox(
    backend: ModalSandbox,
    *,
    csv_path: str,
) -> None:
    """Upload the dataset into the sandbox at ``/work/dataset.csv``.

    Args:
        backend: A live ``ModalSandbox`` backend to upload into.
        csv_path: Host path to the input CSV.
    """
    # Pre-create target directories in sandbox filesystem.
    # modal.Sandbox.open does not automatically create parent directories, so they must exist first.
    await asyncio.to_thread(backend.execute, "mkdir -p /work")

    uploads: list[tuple[str, bytes]] = [("/work/dataset.csv", pathlib.Path(csv_path).read_bytes())]
    await asyncio.to_thread(backend.upload_files, uploads)


async def download_artifacts(
    backend: ModalSandbox,
    *,
    local_root: pathlib.Path,
) -> list[pathlib.Path]:
    """Mirror the sandbox's ``/work/`` output artifacts back onto the host.

    Downloads ``report.md``, ``dataset.clean.csv``, ``changes.json``,
    ``profile.json``, and every PNG under ``/work/plots/``. Files that don't
    exist in the sandbox are skipped silently (e.g. ``changes.json`` and
    ``dataset.clean.csv`` are absent when cleaning was skipped).

    Any pre-existing ``local_root`` is removed before downloading so that
    stale artifacts from a previous run cannot persist and be misinterpreted
    as current results.

    Args:
        backend: A live ``ModalSandbox`` backend to download from.
        local_root: Host directory to mirror the artifacts into. Removed and
            recreated on each call; subdirectories (``plots/``) are created
            on demand.

    Returns:
        Sorted list of host paths that were actually written.
    """
    artifacts = [
        "/work/report.md",
        "/work/dataset.clean.csv",
        "/work/changes.json",
        "/work/profile.json",
    ]

    # Plot filenames are model-chosen; discover them.
    ls = await asyncio.to_thread(backend.execute, "ls -1 /work/plots 2>/dev/null || true")
    for line in ls.output.splitlines():
        name = line.strip()
        if name:
            artifacts.append(f"/work/plots/{name}")

    # Wipe any stale artifacts from a previous run before writing new ones.
    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    written: list[pathlib.Path] = []
    results = await asyncio.to_thread(backend.download_files, artifacts)
    for result in results:
        if result.content is None:
            continue
        rel = pathlib.Path(result.path).relative_to("/work")
        out_path = local_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(result.content)
        written.append(out_path)

    return sorted(written)
