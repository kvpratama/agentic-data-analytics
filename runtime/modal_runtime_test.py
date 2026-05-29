"""Tests for the Modal runtime glue."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

from langchain_modal import ModalSandbox

from runtime.modal_runtime import build_image, download_artifacts, seed_sandbox


def test_build_image_pins_data_science_stack() -> None:
    """build_image() returns a modal.Image with pandas/scipy/matplotlib/seaborn."""
    image = build_image()
    # We can't introspect Modal's internal layers, so just sanity-check the type
    # and that the constructor didn't raise.
    import modal

    assert isinstance(image, modal.Image)


async def test_seed_sandbox_uploads_dataset(tmp_path: pathlib.Path) -> None:
    """seed_sandbox uploads dataset.csv to /work/ and creates the /work dir."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"a,b\n1,2\n")

    backend = MagicMock(spec=ModalSandbox)

    await seed_sandbox(backend, csv_path=str(csv))

    # mkdir is issued for /work before the upload
    mkdir_cmd: str = backend.aexecute.call_args[0][0]
    assert "/work" in mkdir_cmd

    # Single batched upload with the dataset payload preserved
    backend.aupload_files.assert_called_once()
    uploaded = backend.aupload_files.call_args[0][0]
    paths = {p for p, _ in uploaded}
    assert paths == {"/work/dataset.csv"}
    dataset_bytes = next(b for p, b in uploaded if p == "/work/dataset.csv")
    assert dataset_bytes == b"a,b\n1,2\n"


async def test_seed_sandbox_does_not_upload_skills(tmp_path: pathlib.Path) -> None:
    """Skills are served from the host via CompositeBackend, not uploaded here."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"x\n")

    backend = MagicMock(spec=ModalSandbox)
    await seed_sandbox(backend, csv_path=str(csv))

    uploaded = backend.aupload_files.call_args[0][0]
    paths = {p for p, _ in uploaded}
    assert not any(p.startswith("/skills/") for p in paths)


async def test_seed_sandbox_uploads_existing_mirror_artifacts(tmp_path: pathlib.Path) -> None:
    """seed_sandbox uploads every supported file already present in a mirror."""
    mirror = tmp_path / "mirror"
    plots = mirror / "plots"
    plots.mkdir(parents=True)
    (mirror / "dataset.csv").write_bytes(b"raw")
    (mirror / "profile.json").write_bytes(b"{}")
    (mirror / "dataset.clean.csv").write_bytes(b"clean")
    (mirror / "changes.json").write_bytes(b"[]")
    (mirror / "report.md").write_bytes(b"# report")
    (plots / "chart.png").write_bytes(b"png")

    backend = MagicMock(spec=ModalSandbox)

    await seed_sandbox(backend, mirror_root=mirror)

    mkdir_cmd: str = backend.aexecute.call_args[0][0]
    assert "/work" in mkdir_cmd
    assert "/work/plots" in mkdir_cmd

    uploaded = backend.aupload_files.call_args[0][0]
    assert {p for p, _ in uploaded} == {
        "/work/dataset.csv",
        "/work/profile.json",
        "/work/dataset.clean.csv",
        "/work/changes.json",
        "/work/report.md",
        "/work/plots/chart.png",
    }


def _make_dl_result(path: str, content: bytes | None, error: str | None = None) -> MagicMock:
    """Build a download_files-style result object."""
    r = MagicMock()
    r.path = path
    r.content = content
    r.error = error
    return r


async def test_download_artifacts_writes_present_files_to_local_mirror(
    tmp_path: pathlib.Path,
) -> None:
    """Files returned by backend.download_files are written under local_root."""
    backend = MagicMock(spec=ModalSandbox)

    # Sandbox lists three plot files, one with a space
    backend.aexecute.return_value = MagicMock(output="dist_age.png\ncorrelation.png\nmy plot.png\n")

    # Sandbox returns report.md + both plots; changes.json/profile.json are missing
    async def fake_download(paths: list[str]) -> list[MagicMock]:
        bundle = {
            "/work/report.md": b"# report",
            "/work/plots/dist_age.png": b"<png-1>",
            "/work/plots/correlation.png": b"<png-2>",
            "/work/plots/my plot.png": b"<png-3>",
        }
        return [
            _make_dl_result(p, bundle.get(p), None if p in bundle else "not found") for p in paths
        ]

    backend.adownload_files.side_effect = fake_download

    local_root = tmp_path / "out"
    written = await download_artifacts(backend, local_root=local_root)

    assert (local_root / "report.md").read_bytes() == b"# report"
    assert (local_root / "plots" / "dist_age.png").read_bytes() == b"<png-1>"
    assert (local_root / "plots" / "correlation.png").read_bytes() == b"<png-2>"
    assert (local_root / "plots" / "my plot.png").read_bytes() == b"<png-3>"
    # Missing artifacts skipped silently
    assert not (local_root / "changes.json").exists()
    assert not (local_root / "profile.json").exists()

    # Return value lists exactly the files that were written
    assert local_root / "report.md" in written
    assert local_root / "plots" / "dist_age.png" in written
    assert local_root / "plots" / "correlation.png" in written
    assert local_root / "plots" / "my plot.png" in written
    assert len(written) == 4


async def test_download_artifacts_handles_empty_plots_dir(tmp_path: pathlib.Path) -> None:
    """If /work/plots is empty or missing, we still download top-level artifacts."""
    backend = MagicMock(spec=ModalSandbox)
    backend.aexecute.return_value = MagicMock(output="")
    backend.adownload_files.return_value = [
        _make_dl_result("/work/report.md", b"# r"),
        _make_dl_result("/work/changes.json", None, "missing"),
        _make_dl_result("/work/profile.json", None, "missing"),
    ]

    local_root = tmp_path / "out"
    written = await download_artifacts(backend, local_root=local_root)

    assert (local_root / "report.md").read_bytes() == b"# r"
    # Plots dir never created (no plots to write)
    assert not (local_root / "plots").exists()
    assert written == [local_root / "report.md"]


async def test_download_artifacts_preserves_raw_dataset_and_refreshes_plots(
    tmp_path: pathlib.Path,
) -> None:
    """Downloads overwrite artifacts without deleting the raw dataset; plots are refreshed."""
    backend = MagicMock(spec=ModalSandbox)
    backend.aexecute.return_value = MagicMock(output="new.png\n")
    backend.adownload_files.return_value = [
        _make_dl_result("/work/report.md", b"# fresh"),
        _make_dl_result("/work/dataset.clean.csv", None, "missing"),
        _make_dl_result("/work/changes.json", None, "missing"),
        _make_dl_result("/work/profile.json", None, "missing"),
        _make_dl_result("/work/plots/new.png", b"<new>"),
    ]

    local_root = tmp_path / "out"
    local_root.mkdir(parents=True)
    (local_root / "dataset.csv").write_bytes(b"raw")
    (local_root / "report.md").write_bytes(b"# stale report")
    (local_root / "plots").mkdir()
    (local_root / "plots" / "old.png").write_bytes(b"<old>")

    written = await download_artifacts(backend, local_root=local_root)

    assert (local_root / "dataset.csv").read_bytes() == b"raw"
    assert (local_root / "report.md").read_bytes() == b"# fresh"
    assert not (local_root / "plots" / "old.png").exists()
    assert (local_root / "plots" / "new.png").read_bytes() == b"<new>"
    assert written == [local_root / "plots" / "new.png", local_root / "report.md"]


def test_build_image_includes_scikit_learn() -> None:
    """build_image() pins scikit-learn so analyst baselines can run inside the sandbox.

    Modal's Image object doesn't expose the resolved package list publicly, so we
    introspect the recipe via the documented private attribute used by Modal's own
    tests (image._deferred_mounts is not it — fall back to source inspection).
    """
    import inspect

    from runtime import modal_runtime

    source = inspect.getsource(modal_runtime.build_image)
    assert "scikit-learn" in source, (
        "build_image() must pin scikit-learn so the analyst can run predictive baselines"
    )
