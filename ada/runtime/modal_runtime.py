"""Host-side helpers for driving a Modal sandbox.

Three responsibilities:

* :func:`build_image` declares the Modal Image with the data-science stack
  baked in.
* :func:`seed_sandbox` uploads the dataset into the sandbox.
* :func:`download_artifacts` mirrors the sandbox's ``/workspace/`` outputs back
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
    csv_path: str | None = None,
    mirror_root: pathlib.Path | None = None,
) -> None:
    """Upload host-side mirror files into the sandbox under ``/workspace/``.

    Args:
        backend: A live ``ModalSandbox`` backend to upload into.
        csv_path: Host path to the input CSV. Kept for the one-shot CLI-compatible
            call path; uploaded as ``/workspace/dataset.csv``.
        mirror_root: Host mirror directory for a thread. Any supported files
            already present in it are uploaded to their matching ``/workspace/`` paths.

    Raises:
        ValueError: If neither ``csv_path`` nor ``mirror_root`` is provided.
    """
    if csv_path is None and mirror_root is None:
        msg = "seed_sandbox requires csv_path or mirror_root"
        raise ValueError(msg)

    # Pre-create target directories in sandbox filesystem.
    # modal.Sandbox.open does not automatically create parent directories, so they must exist first.
    await backend.aexecute("mkdir -p /workspace /workspace/plots")

    def _collect_uploads() -> list[tuple[str, bytes]]:
        """Collect upload payloads as ``(path, bytes)`` pairs.

        Returns:
            list[tuple[str, bytes]]: Remote destination paths paired with file
            contents to upload into the sandbox.
        """
        uploads: list[tuple[str, bytes]] = []
        if mirror_root is not None:
            top_level = [
                ("dataset.csv", "/workspace/dataset.csv"),
                ("profile.json", "/workspace/profile.json"),
                ("dataset.clean.csv", "/workspace/dataset.clean.csv"),
                ("changes.json", "/workspace/changes.json"),
                ("report.md", "/workspace/report.md"),
            ]
            for local_name, remote_path in top_level:
                source = mirror_root / local_name
                if source.exists():
                    uploads.append((remote_path, source.read_bytes()))

            plots_dir = mirror_root / "plots"
            if plots_dir.exists():
                for plot in sorted(plots_dir.glob("*.png")):
                    uploads.append((f"/workspace/plots/{plot.name}", plot.read_bytes()))
        else:
            uploads.append(("/workspace/dataset.csv", pathlib.Path(str(csv_path)).read_bytes()))
        return uploads

    uploads = await asyncio.to_thread(_collect_uploads)
    if uploads:
        await backend.aupload_files(uploads)


async def download_artifacts(
    backend: ModalSandbox,
    *,
    local_root: pathlib.Path,
) -> list[pathlib.Path]:
    """Mirror the sandbox's ``/workspace/`` output artifacts back onto the host.

    Downloads ``report.md``, ``dataset.clean.csv``, ``changes.json``,
    ``profile.json``, and every PNG under ``/workspace/plots/``. Files that don't
    exist in the sandbox are skipped silently (e.g. ``changes.json`` and
    ``dataset.clean.csv`` are absent when cleaning was skipped).

    Top-level files are overwritten in place so durable inputs such as
    ``dataset.csv`` remain available across turns. The local ``plots/`` directory
    is refreshed before plot downloads so deleted sandbox plots do not linger.

    Args:
        backend: A live ``ModalSandbox`` backend to download from.
        local_root: Host directory to mirror the artifacts into. Created on
            demand; subdirectories (``plots/``) are created as needed.

    Returns:
        Sorted list of host paths that were actually written.
    """
    artifacts = [
        "/workspace/report.md",
        "/workspace/dataset.clean.csv",
        "/workspace/changes.json",
        "/workspace/profile.json",
    ]

    # Plot filenames are model-chosen; discover them.
    ls = await backend.aexecute("ls -1 /workspace/plots 2>/dev/null || true")
    for line in ls.output.splitlines():
        name = line.strip()
        if name:
            artifacts.append(f"/workspace/plots/{name}")

    def _prepare_local_root() -> None:
        """Create ``local_root`` and refresh the ``plots`` directory.

        Returns:
            None: This function does not return a value.
        """
        local_root.mkdir(parents=True, exist_ok=True)
        plots_root = local_root / "plots"
        if plots_root.exists():
            shutil.rmtree(plots_root)

    await asyncio.to_thread(_prepare_local_root)

    results = await backend.adownload_files(artifacts)

    def _write_results() -> list[pathlib.Path]:
        """Write downloaded results into ``local_root``.

        Returns:
            list[pathlib.Path]: Sorted paths for the files written to the host
            mirror.
        """
        written: list[pathlib.Path] = []
        for result in results:
            if result.content is None:
                continue
            rel = pathlib.Path(result.path).relative_to("/workspace")
            out_path = local_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(result.content)
            written.append(out_path)
        return sorted(written)

    return await asyncio.to_thread(_write_results)
