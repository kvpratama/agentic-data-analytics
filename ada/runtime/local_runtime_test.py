"""Tests for local workspace backend."""

import pathlib
import sys

from ada.runtime.local_runtime import LocalWorkspaceBackend


def test_execute_translates_workspace_path(tmp_path: pathlib.Path) -> None:
    """execute() replaces /workspace/ with the real workspace path."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "test.txt").write_text("hello")
    backend = LocalWorkspaceBackend(workspace_path=ws)
    result = backend.execute(f"cat {ws / 'test.txt'}")
    assert result.exit_code == 0, result.output
    # The translation should keep /workspace/ working
    result = backend.execute("cat /workspace/test.txt")
    assert result.exit_code == 0, result.output
    assert "hello" in result.output


def test_execute_works_with_python_one_liners(tmp_path: pathlib.Path) -> None:
    """Python scripts using /workspace/ paths run correctly."""
    ws = tmp_path / "ws"
    ws.mkdir()
    dataset = ws / "dataset.csv"
    dataset.write_text("a,b\n1,2\n")
    backend = LocalWorkspaceBackend(workspace_path=ws)
    cmd = (
        f"{sys.executable} -c "
        "\"with open('/workspace/dataset.csv') as f: "
        'print(len(f.read().splitlines()))"'
    )
    result = backend.execute(cmd)
    assert result.exit_code == 0, result.output
    assert "2" in result.output


def test_execute_does_not_overtranslate_non_workspace(tmp_path: pathlib.Path) -> None:
    """Paths containing /workspace/ as a substring but not at the start are not affected."""
    ws = tmp_path / "ws"
    ws.mkdir()
    backend = LocalWorkspaceBackend(workspace_path=ws)
    result = backend.execute("echo /etc/workspace/config")
    assert result.exit_code == 0
    # Should NOT have rewritten /etc/workspace/ → /etc/ws/workspace/
    assert "/etc/workspace/config" in result.output


def test_read_file_via_virtual_mode(tmp_path: pathlib.Path) -> None:
    """File tools resolve /workspace/ paths through virtual_mode."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "workspace").mkdir()
    (ws / "workspace" / "data.txt").write_text("content")
    backend = LocalWorkspaceBackend(workspace_path=ws)
    result = backend.read("/workspace/data.txt")
    # read() returns ReadResult(error, file_data) — not a .content field
    assert result.error is None
    assert result.file_data is not None
