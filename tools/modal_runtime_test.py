"""Tests for the Modal runtime glue."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

from langchain_modal import ModalSandbox

from tools.modal_runtime import build_image, download_artifacts, seed_sandbox


def test_build_image_pins_data_science_stack():
    """build_image() returns a modal.Image with pandas/scipy/matplotlib/seaborn."""
    image = build_image()
    # We can't introspect Modal's internal layers, so just sanity-check the type
    # and that the constructor didn't raise.
    import modal

    assert isinstance(image, modal.Image)


def test_seed_sandbox_uploads_dataset_and_skills(tmp_path: pathlib.Path):
    """seed_sandbox uploads dataset.csv to /work/ and skills/ tree to /skills/."""
    # Arrange — fake local layout
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"a,b\n1,2\n")

    skills = tmp_path / "skills"
    (skills / "profiler_skills" / "profiler").mkdir(parents=True)
    (skills / "profiler_skills" / "profiler" / "SKILL.md").write_bytes(b"# profiler\n")
    (skills / "cleaner_skills" / "cleaner").mkdir(parents=True)
    (skills / "cleaner_skills" / "cleaner" / "SKILL.md").write_bytes(b"# cleaner\n")

    backend = MagicMock(spec=ModalSandbox)

    # Act
    seed_sandbox(backend, csv_path=str(csv), skills_dir=str(skills))

    # Assert — single batched upload call with the right (path, bytes) tuples
    backend.upload_files.assert_called_once()
    uploaded = backend.upload_files.call_args[0][0]
    paths = {p for p, _ in uploaded}
    assert "/work/dataset.csv" in paths
    assert "/skills/profiler_skills/profiler/SKILL.md" in paths
    assert "/skills/cleaner_skills/cleaner/SKILL.md" in paths

    # Dataset payload preserved
    dataset_bytes = next(b for p, b in uploaded if p == "/work/dataset.csv")
    assert dataset_bytes == b"a,b\n1,2\n"


def test_seed_sandbox_skips_directories_in_skills_tree(tmp_path: pathlib.Path):
    """Only files (not directories) are uploaded from the skills tree."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"x\n")

    skills = tmp_path / "skills"
    (skills / "empty_dir").mkdir(parents=True)
    (skills / "with_file").mkdir()
    (skills / "with_file" / "SKILL.md").write_bytes(b"# x\n")

    backend = MagicMock(spec=ModalSandbox)
    seed_sandbox(backend, csv_path=str(csv), skills_dir=str(skills))

    uploaded = backend.upload_files.call_args[0][0]
    paths = {p for p, _ in uploaded}
    assert "/skills/with_file/SKILL.md" in paths
    assert not any("empty_dir" in p for p in paths)


def _make_dl_result(path: str, content: bytes | None, error: str | None = None):
    """Build a download_files-style result object."""
    r = MagicMock()
    r.path = path
    r.content = content
    r.error = error
    return r


def test_download_artifacts_writes_present_files_to_local_mirror(tmp_path: pathlib.Path):
    """Files returned by backend.download_files are written under local_root."""
    backend = MagicMock(spec=ModalSandbox)

    # Sandbox lists two plot files
    backend.execute.return_value = MagicMock(output="dist_age.png correlation.png\n")

    # Sandbox returns report.md + both plots; changes.json/profile.json are missing
    def fake_download(paths):
        bundle = {
            "/work/report.md": b"# report",
            "/work/plots/dist_age.png": b"<png-1>",
            "/work/plots/correlation.png": b"<png-2>",
        }
        return [
            _make_dl_result(p, bundle.get(p), None if p in bundle else "not found") for p in paths
        ]

    backend.download_files.side_effect = fake_download

    local_root = tmp_path / "out"
    download_artifacts(backend, local_root=local_root)

    assert (local_root / "report.md").read_bytes() == b"# report"
    assert (local_root / "plots" / "dist_age.png").read_bytes() == b"<png-1>"
    assert (local_root / "plots" / "correlation.png").read_bytes() == b"<png-2>"
    # Missing artifacts skipped silently
    assert not (local_root / "changes.json").exists()
    assert not (local_root / "profile.json").exists()


def test_download_artifacts_handles_empty_plots_dir(tmp_path: pathlib.Path):
    """If /work/plots is empty or missing, we still download top-level artifacts."""
    backend = MagicMock(spec=ModalSandbox)
    backend.execute.return_value = MagicMock(output="")
    backend.download_files.return_value = [
        _make_dl_result("/work/report.md", b"# r"),
        _make_dl_result("/work/changes.json", None, "missing"),
        _make_dl_result("/work/profile.json", None, "missing"),
    ]

    local_root = tmp_path / "out"
    download_artifacts(backend, local_root=local_root)

    assert (local_root / "report.md").read_bytes() == b"# r"
    # Plots dir never created (no plots to write)
    assert not (local_root / "plots").exists()
