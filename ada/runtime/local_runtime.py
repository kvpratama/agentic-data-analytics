"""Local backend with /workspace/ path translation for execute commands."""

from __future__ import annotations

import asyncio
import pathlib
import shutil

from deepagents.backends.local_shell import ExecuteResponse, LocalShellBackend

# Top-level artifacts mirrored from <root>/workspace/ to the mirror root.
_ARTIFACT_NAMES = ("report.md", "dataset.clean.csv", "changes.json", "profile.json")


class LocalWorkspaceBackend(LocalShellBackend):
    """LocalShellBackend that maps /workspace/ to a real on-disk path in execute commands.

    All subagent skills use hardcoded ``/workspace/`` paths in both file tools and
    ``execute`` commands. File tools are handled by ``virtual_mode=True`` (``/workspace/``
    maps to ``<root>/workspace/``). For ``execute``, we rewrite ``/workspace/`` to the
    real path before delegating to ``subprocess.run``.
    """

    def __init__(
        self,
        workspace_path: pathlib.Path,
        *,
        timeout: int = 120,
        max_output_bytes: int = 100_000,
    ) -> None:
        """Initialize.

        Args:
            workspace_path: Real host directory that acts as the workspace root.
                A ``workspace/`` subdirectory will be created inside it.
            timeout: Default execute timeout.
            max_output_bytes: Max output capture size.
        """
        self.workspace_path = pathlib.Path(workspace_path).resolve()
        super().__init__(
            root_dir=str(self.workspace_path),
            virtual_mode=True,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            inherit_env=True,
            # env={"PATH": str(self.workspace_path / ".." / ".." / ".venv" / "bin")},
        )

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute with /workspace/ translated to the real workspace path."""
        import re

        # Translate /workspace/ only if it's not preceded by path characters
        # (e.g., avoid overtranslating /etc/workspace/ or ../workspace/)
        translated = re.sub(
            r"(?<![a-zA-Z0-9._~-])/workspace/", f"{self.workspace_path}/workspace/", command
        )
        return super().execute(translated, timeout=timeout)


async def mirror_local_artifacts(
    backend: LocalWorkspaceBackend,
    *,
    local_root: pathlib.Path,
) -> list[pathlib.Path]:
    """Mirror a local backend's ``/workspace/`` outputs up to the mirror root.

    The local backend writes artifacts to ``<root>/workspace/`` (via
    ``virtual_mode``). This copies the same artifact set that
    :func:`ada.runtime.modal_runtime.download_artifacts` produces — ``report.md``,
    ``dataset.clean.csv``, ``changes.json``, ``profile.json``, and every file
    under ``plots/`` — to the top level of ``local_root`` so the CLI and
    Modal-backed runs share an identical host layout. Missing artifacts are
    skipped silently and the top-level ``plots/`` directory is refreshed.

    Args:
        backend: The local backend whose workspace holds the artifacts.
        local_root: Host directory to mirror the artifacts into (the mirror root).

    Returns:
        Sorted list of host paths that were actually written.
    """
    source_root = backend.workspace_path / "workspace"

    def _mirror() -> list[pathlib.Path]:
        """Copy artifacts synchronously and return the written host paths."""
        local_root.mkdir(parents=True, exist_ok=True)
        written: list[pathlib.Path] = []

        for name in _ARTIFACT_NAMES:
            src = source_root / name
            if src.is_file():
                dest = local_root / name
                shutil.copyfile(src, dest)
                written.append(dest)

        dest_plots = local_root / "plots"
        if dest_plots.exists():
            shutil.rmtree(dest_plots)
        src_plots = source_root / "plots"
        if src_plots.is_dir():
            for plot in src_plots.iterdir():
                if plot.is_file():
                    dest_plots.mkdir(parents=True, exist_ok=True)
                    dest = dest_plots / plot.name
                    shutil.copyfile(plot, dest)
                    written.append(dest)

        return sorted(written)

    return await asyncio.to_thread(_mirror)
