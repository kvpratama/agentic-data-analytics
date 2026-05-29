"""Tests for workspace mirror and sandbox provisioning logic."""

import pathlib
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_modal import ModalSandbox

from config import Settings
from runtime.workspace import (
    SandboxResources,
    bootstrap_mirror,
    create_sandbox,
    get_mirror_root,
    provision_workspace,
)


async def _run_to_thread_sync[**P, T](
    func: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    return func(*args, **kwargs)


def test_bootstrap_mirror_copies_dataset_once(tmp_path: pathlib.Path) -> None:
    """Mirror bootstrap creates dataset.csv and does not overwrite it later."""
    source = tmp_path / "source.csv"
    source.write_bytes(b"a\n1\n")
    mirror = tmp_path / "workspace" / "source_thread"

    bootstrap_mirror(mirror, source)
    assert (mirror / "dataset.csv").read_bytes() == b"a\n1\n"

    source.write_bytes(b"a\n2\n")
    bootstrap_mirror(mirror, source)
    assert (mirror / "dataset.csv").read_bytes() == b"a\n1\n"


def test_get_mirror_root(tmp_path: pathlib.Path) -> None:
    """get_mirror_root generates a safe path under workspace/."""
    with patch("runtime.workspace._project_root", return_value=tmp_path):
        root = get_mirror_root("dataset", "thread-123")
    assert root == tmp_path / "workspace" / "dataset_thread-123"


def test_get_mirror_root_validates_components() -> None:
    """get_mirror_root rejects unsafe strings."""
    with pytest.raises(ValueError, match="invalid stem"):
        get_mirror_root("../dataset", "thread")
    with pytest.raises(ValueError, match="invalid thread_id"):
        get_mirror_root("dataset", "thread/123")


async def test_create_sandbox_creates_modal_sandbox() -> None:
    """create_sandbox initializes the Modal sandbox and app."""
    settings = Settings(
        _env_file=None,  # type: ignore
        modal_app_name="test-app",
        modal_sandbox_timeout=600,
    )
    sandbox = MagicMock(name="Sandbox")
    sandbox.terminate.aio = AsyncMock()
    app = MagicMock(name="App")

    with (
        patch("runtime.workspace.get_settings", return_value=settings),
        patch(
            "runtime.workspace.modal.App.lookup.aio", new=AsyncMock(return_value=app)
        ) as mock_app,
        patch(
            "runtime.workspace.modal.Sandbox.create.aio", new=AsyncMock(return_value=sandbox)
        ) as mock_create,
        patch("runtime.workspace.build_image", return_value="fake_image"),
    ):
        resources = await create_sandbox("thread-1")

    mock_app.assert_awaited_once_with("test-app", create_if_missing=True)
    mock_create.assert_awaited_once_with(
        image="fake_image",
        app=app,
        tags={"thread_id": "thread-1"},
        timeout=600,
        cpu=2.0,
        memory=4096,
    )
    assert isinstance(resources.backend, ModalSandbox)
    assert resources.terminate is sandbox.terminate.aio


async def test_provision_workspace_creates_fresh_sandbox_and_seeds_first_turn(
    tmp_path: pathlib.Path,
) -> None:
    """provision_workspace bootstraps a thread mirror and seeds a fresh Modal sandbox."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    backend = MagicMock(spec=ModalSandbox)
    terminate = AsyncMock()
    resources = SandboxResources(backend=backend, terminate=terminate)

    with (
        patch("runtime.workspace._project_root", return_value=tmp_path),
        patch("runtime.workspace.asyncio.to_thread", side_effect=_run_to_thread_sync),
        patch("runtime.workspace.create_sandbox", new=AsyncMock(return_value=resources)) as create,
        patch("runtime.workspace.seed_sandbox", new=AsyncMock()) as seed,
    ):
        res, mirror_root = await provision_workspace("input", "thread-1", csv)

    mirror = tmp_path / "workspace" / "input_thread-1"
    assert res is resources
    assert mirror_root == mirror
    assert (mirror / "dataset.csv").read_bytes() == b"a,b\n1,2\n"
    create.assert_awaited_once_with("thread-1")
    seed.assert_awaited_once_with(backend, mirror_root=mirror)


async def test_provision_workspace_terminates_sandbox_if_seeding_fails(
    tmp_path: pathlib.Path,
) -> None:
    """provision_workspace cleans up the sandbox if seeding raises before returning."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"a,b\n1,2\n")
    backend = MagicMock(name="ModalSandbox")
    terminate = AsyncMock()
    resources = SandboxResources(backend=backend, terminate=terminate)

    with (
        patch("runtime.workspace._project_root", return_value=tmp_path),
        patch("runtime.workspace.asyncio.to_thread", side_effect=_run_to_thread_sync),
        patch("runtime.workspace.create_sandbox", new=AsyncMock(return_value=resources)),
        patch(
            "runtime.workspace.seed_sandbox",
            new=AsyncMock(side_effect=RuntimeError("seed failed")),
        ),
    ):
        with pytest.raises(RuntimeError, match="seed failed"):
            await provision_workspace("input", "thread-1", csv)

    terminate.assert_awaited_once()


async def test_provision_workspace_reuploads_existing_mirror_on_followup(
    tmp_path: pathlib.Path,
) -> None:
    """provision_workspace keeps thread workspace continuity by seeding accumulated files."""
    csv = tmp_path / "input.csv"
    csv.write_bytes(b"raw")
    mirror = tmp_path / "workspace" / "input_thread-1"
    plots = mirror / "plots"
    plots.mkdir(parents=True)
    (mirror / "dataset.csv").write_bytes(b"raw")
    (mirror / "profile.json").write_bytes(b"{}")
    (mirror / "dataset.clean.csv").write_bytes(b"clean")
    (plots / "chart.png").write_bytes(b"png")

    backend = MagicMock(spec=ModalSandbox)
    terminate = AsyncMock()
    resources = SandboxResources(backend=backend, terminate=terminate)

    with (
        patch("runtime.workspace._project_root", return_value=tmp_path),
        patch("runtime.workspace.asyncio.to_thread", side_effect=_run_to_thread_sync),
        patch("runtime.workspace.create_sandbox", new=AsyncMock(return_value=resources)) as create,
        patch("runtime.workspace.seed_sandbox", new=AsyncMock()) as seed,
    ):
        await provision_workspace("input", "thread-1", csv)

    create.assert_awaited_once()
    seed.assert_awaited_once_with(backend, mirror_root=mirror)
