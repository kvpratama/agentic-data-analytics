"""Host-side helpers for driving a Modal sandbox.

Three responsibilities:

* :func:`build_image` declares the Modal Image with the data-science stack
  baked in.
* :func:`seed_sandbox` uploads the dataset and skills tree into the sandbox.
* :func:`download_artifacts` mirrors the sandbox's ``/work/`` outputs back
  onto the host so users can open ``report.md`` and the plots locally.

All Modal-facing calls go through the ``ModalSandbox`` backend instance the
caller passes in — these helpers do not own the sandbox's lifecycle.
"""

from __future__ import annotations

import pathlib

import modal
from langchain_modal import ModalSandbox


def build_image() -> modal.Image:
    """Return the Modal Image used for every sandbox in this project.

    Pins the data-science packages the subagents need. Modal caches the built
    image after the first run, so subsequent sandbox creations are fast.

    Returns:
        A configured ``modal.Image`` ready to pass to ``modal.Sandbox.create``.
    """
    return modal.Image.debian_slim(python_version="3.12").pip_install(
        "pandas>=3.0",
        "scipy>=1.17",
        "matplotlib>=3.10",
        "seaborn>=0.13",
    )


def seed_sandbox(
    backend: ModalSandbox,
    *,
    csv_path: str,
    skills_dir: str,
) -> None:
    """Upload the dataset and skills tree into the sandbox.

    The dataset lands at ``/work/dataset.csv``. Every file under ``skills_dir``
    is mirrored to ``/skills/<relative-path>`` preserving the directory
    layout so ``SkillsMiddleware`` can discover SKILL.md files.

    Args:
        backend: A live ``ModalSandbox`` backend to upload into.
        csv_path: Host path to the input CSV.
        skills_dir: Host path to the project's ``skills/`` directory.
    """
    uploads: list[tuple[str, bytes]] = [("/work/dataset.csv", pathlib.Path(csv_path).read_bytes())]

    skills_root = pathlib.Path(skills_dir)
    for entry in skills_root.rglob("*"):
        if entry.is_file():
            rel = entry.relative_to(skills_root).as_posix()
            uploads.append((f"/skills/{rel}", entry.read_bytes()))

    backend.upload_files(uploads)


def download_artifacts(
    backend: ModalSandbox,
    *,
    local_root: pathlib.Path,
) -> None:
    """Mirror the sandbox's ``/work/`` output artifacts back onto the host.

    Downloads ``report.md``, ``changes.json``, ``profile.json``, and every PNG
    under ``/work/plots/``. Files that don't exist in the sandbox are skipped
    silently (e.g. ``changes.json`` is absent when cleaning was skipped).

    Args:
        backend: A live ``ModalSandbox`` backend to download from.
        local_root: Host directory to mirror the artifacts into. Created if
            missing; subdirectories (``plots/``) are created on demand.
    """
    artifacts = ["/work/report.md", "/work/changes.json", "/work/profile.json"]

    # Plot filenames are model-chosen; discover them.
    ls = backend.execute("ls /work/plots 2>/dev/null || true")
    for name in ls.output.split():
        artifacts.append(f"/work/plots/{name}")

    local_root.mkdir(parents=True, exist_ok=True)

    for result in backend.download_files(artifacts):
        if result.content is None:
            continue
        rel = pathlib.Path(result.path).relative_to("/work")
        out_path = local_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(result.content)
