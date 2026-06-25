"""Tests for local workspace backend."""

import pathlib
import sys

from ada.runtime.local_runtime import LocalWorkspaceBackend, mirror_local_artifacts


def test_execute_translates_workspace_path(tmp_path: pathlib.Path) -> None:
    """execute() replaces /workspace/ with the real workspace path."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "workspace").mkdir()
    (ws / "workspace" / "test.txt").write_text("hello")
    backend = LocalWorkspaceBackend(workspace_path=ws)
    result = backend.execute(f"cat {ws / 'workspace' / 'test.txt'}")
    assert result.exit_code == 0, result.output
    # The translation should keep /workspace/ working
    result = backend.execute("cat /workspace/test.txt")
    assert result.exit_code == 0, result.output
    assert "hello" in result.output


def test_execute_works_with_python_one_liners(tmp_path: pathlib.Path) -> None:
    """Python scripts using /workspace/ paths run correctly."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "workspace").mkdir()
    dataset = ws / "workspace" / "dataset.csv"
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


async def test_mirror_local_artifacts_copies_workspace_outputs_to_root(
    tmp_path: pathlib.Path,
) -> None:
    """Artifacts under <root>/workspace/ are mirrored to the top-level mirror root."""
    root = tmp_path / "mirror"
    ws = root / "workspace"
    (ws / "plots").mkdir(parents=True)
    (ws / "report.md").write_text("# report")
    (ws / "profile.json").write_text("{}")
    (ws / "plots" / "dist.png").write_bytes(b"<png>")
    # An unrelated workspace file is not mirrored.
    (ws / "scratch.txt").write_text("ignore me")
    backend = LocalWorkspaceBackend(workspace_path=root)

    written = await mirror_local_artifacts(backend, local_root=root)

    assert (root / "report.md").read_text() == "# report"
    assert (root / "profile.json").read_text() == "{}"
    assert (root / "plots" / "dist.png").read_bytes() == b"<png>"
    assert not (root / "scratch.txt").exists()
    assert written == [
        root / "plots" / "dist.png",
        root / "profile.json",
        root / "report.md",
    ]


async def test_mirror_local_artifacts_refreshes_plots_and_skips_missing(
    tmp_path: pathlib.Path,
) -> None:
    """Stale top-level plots are cleared and missing artifacts are skipped silently."""
    root = tmp_path / "mirror"
    ws = root / "workspace"
    (ws / "plots").mkdir(parents=True)
    (ws / "plots" / "new.png").write_bytes(b"<new>")
    # Pre-existing top-level state: raw dataset preserved, stale plot removed.
    (root / "dataset.csv").write_bytes(b"raw")
    (root / "plots").mkdir()
    (root / "plots" / "old.png").write_bytes(b"<old>")
    backend = LocalWorkspaceBackend(workspace_path=root)

    written = await mirror_local_artifacts(backend, local_root=root)

    assert (root / "dataset.csv").read_bytes() == b"raw"
    assert not (root / "plots" / "old.png").exists()
    assert (root / "plots" / "new.png").read_bytes() == b"<new>"
    assert not (root / "report.md").exists()
    assert written == [root / "plots" / "new.png"]
