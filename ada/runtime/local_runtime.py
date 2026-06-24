"""Local backend with /workspace/ path translation for execute commands."""

from __future__ import annotations

import pathlib

from deepagents.backends.local_shell import ExecuteResponse, LocalShellBackend


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
        translated = re.sub(r"(?<![a-zA-Z0-9._~-])/workspace/", f"{self.workspace_path}/", command)
        return super().execute(translated, timeout=timeout)
