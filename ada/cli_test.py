"""Tests for the ada CLI/REPL."""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import time
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console as RealConsole

from ada.cli import (
    ExitRepl,
    Session,
    SlashError,
    _amain,
    _cmd_exit,
    _cmd_help,
    _safe_stem,
    build_arg_parser,
    choose_csv,
    discover_csvs,
    dispatch_slash,
    list_thread_dirs,
    open_report,
    parse_command,
    parse_thread_id,
    repl_loop,
    run_agent_turn,
)


def test_discover_csvs_empty_dir(tmp_path: pathlib.Path) -> None:
    """No CSVs in the directory returns an empty list."""
    assert discover_csvs(tmp_path) == []


def test_discover_csvs_returns_sorted_csv_files(tmp_path: pathlib.Path) -> None:
    """CSVs are returned in sorted order, top-level only."""
    (tmp_path / "b.csv").touch()
    (tmp_path / "a.csv").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.csv").touch()
    (tmp_path / "notes.txt").touch()

    found = discover_csvs(tmp_path)

    assert [p.name for p in found] == ["a.csv", "b.csv"]


def test_choose_csv_auto_selects_single(tmp_path: pathlib.Path) -> None:
    """When exactly one CSV is present, it is returned without prompting."""
    only = tmp_path / "only.csv"
    only.touch()
    console = MagicMock()

    result = choose_csv([only], console=console)

    assert result == only


def test_choose_csv_returns_none_for_empty_list() -> None:
    """Zero CSVs returns None; caller is responsible for the user message."""
    console = MagicMock()
    assert choose_csv([], console=console) is None


def test_choose_csv_uses_picker_for_multiple(tmp_path: pathlib.Path) -> None:
    """With multiple CSVs the user is prompted for a 1-based index."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.touch()
    b.touch()
    console = MagicMock()

    with patch("ada.cli.input", return_value="2", create=True):
        result = choose_csv([a, b], console=console)

    assert result == b


def test_choose_csv_picker_rejects_out_of_range(tmp_path: pathlib.Path) -> None:
    """Invalid input re-prompts until a valid number is supplied."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.touch()
    b.touch()
    console = MagicMock()

    with patch("ada.cli.input", side_effect=["9", "0", "abc", "1"], create=True):
        result = choose_csv([a, b], console=console)

    assert result == a


def test_session_stem_derived_from_csv_path(tmp_path: pathlib.Path) -> None:
    """Session.stem is derived from csv_path.stem, not stored separately."""
    csv = tmp_path / "Titanic-Dataset.csv"
    csv.touch()
    s = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="abc",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    assert s.stem == "Titanic-Dataset"


def test_session_stem_returns_none_when_no_csv(tmp_path: pathlib.Path) -> None:
    """A session without a CSV reports stem as None."""
    s = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="abc",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    assert s.stem is None


def test_parse_thread_id_simple() -> None:
    """The substring after the final underscore is the thread_id."""
    assert parse_thread_id("Titanic_abc-123") == "abc-123"


def test_parse_thread_id_with_underscores_in_stem() -> None:
    """A stem containing underscores still yields the trailing component."""
    assert parse_thread_id("housing_data_xyz-456") == "xyz-456"


def test_parse_thread_id_no_underscore_returns_none() -> None:
    """A dirname with no underscore cannot be parsed."""
    assert parse_thread_id("orphan") is None


def test_list_thread_dirs_filters_by_stem(tmp_path: pathlib.Path) -> None:
    """Only directories matching ``<stem>_*`` are returned, sorted by mtime desc."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    a = workspace / "Titanic_aaa"
    b = workspace / "Titanic_bbb"
    other = workspace / "Iris_ccc"
    for d in (a, b, other):
        d.mkdir()
    older = time.time() - 100
    os.utime(a, (older, older))

    result = list_thread_dirs(workspace, stem="Titanic")

    assert result == [b, a]
    assert other not in result


def test_list_thread_dirs_missing_workspace(tmp_path: pathlib.Path) -> None:
    """An absent workspace directory returns an empty list rather than raising."""
    assert list_thread_dirs(tmp_path / "does-not-exist", stem="Titanic") == []


def test_list_thread_dirs_ignores_files(tmp_path: pathlib.Path) -> None:
    """Files matching the pattern are ignored; only directories count."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Titanic_real").mkdir()
    (workspace / "Titanic_file").write_text("not a dir")

    result = list_thread_dirs(workspace, stem="Titanic")

    assert [p.name for p in result] == ["Titanic_real"]


def test_open_report_missing_prints_notice(tmp_path: pathlib.Path) -> None:
    """When the report does not exist, a notice is printed and no command runs."""
    console = MagicMock()
    with patch("ada.cli.subprocess.run") as run:
        open_report(tmp_path / "absent.md", console=console)
    run.assert_not_called()
    console.print.assert_called_once()
    assert "not found" in console.print.call_args.args[0].lower()


def test_open_report_linux_uses_xdg_open(tmp_path: pathlib.Path) -> None:
    """On Linux the helper invokes xdg-open."""
    report = tmp_path / "report.md"
    report.write_text("# r")
    console = MagicMock()
    with (
        patch("ada.cli.sys.platform", "linux"),
        patch("ada.cli.subprocess.run") as run,
    ):
        open_report(report, console=console)
    run.assert_called_once_with(["xdg-open", str(report)], check=False)


def test_open_report_macos_uses_open(tmp_path: pathlib.Path) -> None:
    """On macOS the helper invokes the open(1) command."""
    report = tmp_path / "report.md"
    report.write_text("# r")
    console = MagicMock()
    with (
        patch("ada.cli.sys.platform", "darwin"),
        patch("ada.cli.subprocess.run") as run,
    ):
        open_report(report, console=console)
    run.assert_called_once_with(["open", str(report)], check=False)


def test_open_report_handles_missing_opener_binary(tmp_path: pathlib.Path) -> None:
    """If xdg-open / open is not installed, print a hint instead of crashing."""
    report = tmp_path / "report.md"
    report.write_text("# r")
    console = MagicMock()
    with (
        patch("ada.cli.sys.platform", "linux"),
        patch("ada.cli.subprocess.run", side_effect=FileNotFoundError("xdg-open")),
    ):
        open_report(report, console=console)
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "xdg-open" in rendered or "could not" in rendered.lower()


def test_open_report_windows_uses_startfile(tmp_path: pathlib.Path) -> None:
    """On Windows the helper uses os.startfile."""
    report = tmp_path / "report.md"
    report.write_text("# r")
    console = MagicMock()
    fake_startfile = MagicMock()
    with (
        patch("ada.cli.sys.platform", "win32"),
        patch.object(os, "startfile", fake_startfile, create=True),
    ):
        open_report(report, console=console)
    fake_startfile.assert_called_once_with(str(report))


def test_parse_command_simple() -> None:
    """A basic slash command parses to verb and empty args."""
    assert parse_command("/help") == ("help", [])


def test_parse_command_with_args() -> None:
    """Command args are split on shell-style whitespace."""
    assert parse_command("/csv data.csv") == ("csv", ["data.csv"])


def test_parse_command_quoted_path_with_spaces() -> None:
    """Quoted paths survive parsing."""
    assert parse_command('/csv "my data.csv"') == ("csv", ["my data.csv"])


def test_parse_command_not_a_slash_command() -> None:
    """Plain text is not treated as a slash command."""
    assert parse_command("regular text") is None


def test_parse_command_only_slash() -> None:
    """A bare `/` is not a slash command."""
    assert parse_command("/") is None


async def test_dispatch_help_prints_all_verbs(tmp_path: pathlib.Path) -> None:
    """Help renders a Rich Table that contains every command verb."""
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    await dispatch_slash("help", [], session)
    table = console.print.call_args.args[0]
    sink = StringIO()
    RealConsole(file=sink, force_terminal=False, width=200).print(table)
    rendered = sink.getvalue()
    for verb in ("/help", "/exit", "/new", "/csv", "/list", "/resume", "/open"):
        assert verb in rendered, f"missing {verb} in help output"


async def test_dispatch_exit_raises_exit_repl(tmp_path: pathlib.Path) -> None:
    """The exit command terminates the REPL through a testable exception."""
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with pytest.raises(ExitRepl):
        await dispatch_slash("exit", [], session)


def test_command_handlers_use_google_style_docstrings() -> None:
    """Affected slash handlers describe args and raises using Google-style sections."""
    help_doc = _cmd_help.__doc__
    exit_doc = _cmd_exit.__doc__

    assert help_doc is not None
    assert "Args:" in help_doc
    assert "args:" in help_doc
    assert "session:" in help_doc

    assert exit_doc is not None
    assert "Args:" in exit_doc
    assert "args:" in exit_doc
    assert "session:" in exit_doc
    assert "Raises:" in exit_doc
    assert "ExitRepl:" in exit_doc


async def test_dispatch_new_rotates_thread_id(tmp_path: pathlib.Path) -> None:
    """New starts a fresh thread without switching CSV."""
    csv = tmp_path / "x.csv"
    csv.touch()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="OLD",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    await dispatch_slash("new", [], session)
    assert session.thread_id != "OLD"
    assert session.csv_path == csv


async def test_dispatch_csv_switches_and_rotates(tmp_path: pathlib.Path) -> None:
    """CSV command switches files and starts a new thread."""
    old = tmp_path / "old.csv"
    old.touch()
    new = tmp_path / "new.csv"
    new.touch()
    session = Session(
        cwd=tmp_path,
        csv_path=old,
        thread_id="OLD",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    await dispatch_slash("csv", [str(new)], session)
    assert session.csv_path == new.resolve()
    assert session.thread_id != "OLD"


async def test_dispatch_csv_errors_when_path_missing(tmp_path: pathlib.Path) -> None:
    """Missing CSV paths are reported as slash errors."""
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with pytest.raises(SlashError, match="not a file"):
        await dispatch_slash("csv", [str(tmp_path / "absent.csv")], session)


async def test_dispatch_csv_requires_argument(tmp_path: pathlib.Path) -> None:
    """CSV command requires exactly one path."""
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with pytest.raises(SlashError, match="usage"):
        await dispatch_slash("csv", [], session)


async def test_dispatch_list_prints_cwd_csvs(tmp_path: pathlib.Path) -> None:
    """List prints CSV files from the launch directory."""
    (tmp_path / "a.csv").touch()
    (tmp_path / "b.csv").touch()
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    await dispatch_slash("list", [], session)
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "a.csv" in rendered
    assert "b.csv" in rendered


async def test_dispatch_resume_with_id_jumps_directly(tmp_path: pathlib.Path) -> None:
    """Resume can jump directly to a known thread id."""
    csv = tmp_path / "T.csv"
    csv.touch()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "T_abc-1").mkdir()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="OLD",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with patch("ada.cli.workspace_root", return_value=workspace):
        await dispatch_slash("resume", ["abc-1"], session)
    assert session.thread_id == "abc-1"


async def test_dispatch_resume_unknown_id_errors(tmp_path: pathlib.Path) -> None:
    """Unknown resume ids are user-facing errors."""
    csv = tmp_path / "T.csv"
    csv.touch()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="OLD",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with (
        patch("ada.cli.workspace_root", return_value=workspace),
        pytest.raises(SlashError, match="no thread"),
    ):
        await dispatch_slash("resume", ["nope"], session)


async def test_dispatch_resume_no_args_picks_from_list(tmp_path: pathlib.Path) -> None:
    """Resume opens a picker when no thread id is supplied."""
    csv = tmp_path / "T.csv"
    csv.touch()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "T_one").mkdir()
    (workspace / "T_two").mkdir()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="OLD",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with (
        patch("ada.cli.workspace_root", return_value=workspace),
        patch("ada.cli.input", return_value="1", create=True),
    ):
        await dispatch_slash("resume", [], session)
    assert session.thread_id in {"one", "two"}


async def test_dispatch_resume_requires_csv(tmp_path: pathlib.Path) -> None:
    """Resume requires an active CSV stem."""
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with pytest.raises(SlashError, match="no csv"):
        await dispatch_slash("resume", [], session)


async def test_dispatch_open_report_with_csv(tmp_path: pathlib.Path) -> None:
    """Open report targets the current thread's report path."""
    csv = tmp_path / "T.csv"
    csv.touch()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mirror = workspace / "T_abc"
    mirror.mkdir()
    (mirror / "report.md").write_text("# r")
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="abc",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with (
        patch("ada.cli.workspace_root", return_value=workspace),
        patch("ada.cli.open_report") as opener,
    ):
        await dispatch_slash("open", ["report"], session)
    opener.assert_called_once_with(mirror / "report.md", console=session.console)


async def test_dispatch_open_unknown_target(tmp_path: pathlib.Path) -> None:
    """Only /open report is currently supported."""
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with pytest.raises(SlashError, match="usage"):
        await dispatch_slash("open", ["unknown"], session)


async def test_dispatch_unknown_verb(tmp_path: pathlib.Path) -> None:
    """Unknown command names become SlashError."""
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    with pytest.raises(SlashError, match="unknown"):
        await dispatch_slash("blah", [], session)


async def test_run_agent_turn_errors_without_csv(tmp_path: pathlib.Path) -> None:
    """A user message with no CSV selected gets a hint instead of crashing."""
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    await run_agent_turn(session, "what columns are there?")
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "csv" in rendered.lower()


async def test_run_agent_turn_streams_chunks(tmp_path: pathlib.Path) -> None:
    """Happy path: agent is built with the session's checkpointer + thread_id."""
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    console = MagicMock()
    checkpointer = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="thr-1",
        checkpointer=checkpointer,
        console=console,
    )

    async def fake_astream(_inputs, config, **kwargs):  # noqa: ANN001
        assert config["configurable"]["thread_id"] == "thr-1"
        msg = MagicMock(content="hello", name="assistant")
        yield {"type": "updates", "data": {"model": {"messages": [msg]}}}

    fake_graph = MagicMock()
    fake_graph.astream = fake_astream
    backend = MagicMock()
    terminate = AsyncMock()
    from ada.runtime.workspace import SandboxResources

    resources = SandboxResources(backend=backend, terminate=terminate)
    mirror_root = tmp_path / "mirror"

    with (
        patch(
            "ada.cli.provision_workspace",
            new=AsyncMock(return_value=(resources, mirror_root)),
        ) as provision,
        patch("ada.cli.create_analytics_agent", return_value=fake_graph) as create,
    ):
        await run_agent_turn(session, "summarize the data")

    provision.assert_awaited_once_with("data", "thr-1", csv.resolve())
    create.assert_called_once_with(
        backend,
        mirror_root=mirror_root,
        terminate_sandbox=terminate,
        checkpointer=checkpointer,
    )
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "hello" in rendered


async def test_run_agent_turn_catches_agent_errors(tmp_path: pathlib.Path) -> None:
    """An agent exception is caught and printed; control returns normally."""
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="thr-1",
        checkpointer=MagicMock(),
        console=console,
    )

    async def boom(_inputs, config=None, **_kwargs):  # noqa: ANN001
        raise RuntimeError("boom")
        yield

    fake_graph = MagicMock()
    fake_graph.astream = boom
    backend = MagicMock()
    terminate = AsyncMock()
    from ada.runtime.workspace import SandboxResources

    resources = SandboxResources(backend=backend, terminate=terminate)

    with (
        patch(
            "ada.cli.provision_workspace", new=AsyncMock(return_value=(resources, tmp_path / "m"))
        ),
        patch("ada.cli.create_analytics_agent", return_value=fake_graph),
    ):
        await run_agent_turn(session, "x")

    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "boom" in rendered or "error" in rendered.lower()


async def test_repl_loop_exits_on_eof(tmp_path: pathlib.Path) -> None:
    """Ctrl-D at the prompt cleanly ends the loop."""
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=EOFError)
    with patch("ada.cli.PromptSession", return_value=fake_session):
        await repl_loop(session)


async def test_repl_loop_runs_slash_then_exit(tmp_path: pathlib.Path) -> None:
    """A /help line dispatches; then /exit terminates."""
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=["/help", "/exit"])
    with patch("ada.cli.PromptSession", return_value=fake_session):
        await repl_loop(session)
    assert console.print.called


async def test_repl_loop_routes_plain_text_to_agent(tmp_path: pathlib.Path) -> None:
    """Non-slash input is routed to run_agent_turn with the literal text."""
    csv = tmp_path / "d.csv"
    csv.touch()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="t",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=["analyze this", EOFError])
    with (
        patch("ada.cli.PromptSession", return_value=fake_session),
        patch("ada.cli.run_agent_turn", new=AsyncMock()) as turn,
    ):
        await repl_loop(session)
    turn.assert_awaited_once_with(session, "analyze this")


async def test_repl_loop_handles_keyboard_interrupt(tmp_path: pathlib.Path) -> None:
    """First Ctrl-C prints a hint and loops; second exits via EOF."""
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=[KeyboardInterrupt, EOFError])
    with patch("ada.cli.PromptSession", return_value=fake_session):
        await repl_loop(session)
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "exit" in rendered.lower() or "ctrl" in rendered.lower()


async def test_repl_loop_surfaces_slash_errors(tmp_path: pathlib.Path) -> None:
    """SlashError is caught and printed; the loop continues."""
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=["/csv", "/exit"])
    with patch("ada.cli.PromptSession", return_value=fake_session):
        await repl_loop(session)
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "usage" in rendered.lower()


def test_arg_parser_defaults_to_interactive() -> None:
    """No args launches interactive mode."""
    args = build_arg_parser().parse_args([])
    assert args.csv is None
    assert args.prompt is None


def test_arg_parser_one_shot() -> None:
    """--csv plus -p selects one-shot mode."""
    args = build_arg_parser().parse_args(["--csv", "data.csv", "-p", "what columns?"])
    assert args.csv == "data.csv"
    assert args.prompt == "what columns?"


def test_arg_parser_legacy_positional() -> None:
    """Positional args select legacy one-shot mode."""
    args = build_arg_parser().parse_args(["data.csv", "what columns?"])
    assert args.csv_pos == "data.csv"
    assert args.prompt_pos == "what columns?"


async def test_amain_interactive_calls_repl(tmp_path: pathlib.Path) -> None:
    """No --csv / -p routes to REPL loop with auto-discovered CSV."""
    (tmp_path / "only.csv").touch()
    args = build_arg_parser().parse_args([])

    fake_checkpointer = MagicMock()

    @contextlib.asynccontextmanager
    async def fake_ctx(_path: str):
        yield fake_checkpointer

    with (
        patch("ada.cli.Path.cwd", return_value=tmp_path),
        patch("ada.cli.load_environment") as load_env,
        patch("ada.cli.AsyncSqliteSaver.from_conn_string", fake_ctx),
        patch("ada.cli.repl_loop", new=AsyncMock()) as repl,
        patch("ada.cli.run_agent_turn", new=AsyncMock()) as turn,
    ):
        await _amain(args)

    load_env.assert_called_once()
    repl.assert_awaited_once()
    turn.assert_not_called()


async def test_amain_one_shot_calls_run_agent_turn(tmp_path: pathlib.Path) -> None:
    """--csv + -p runs a single turn and skips the REPL."""
    csv = tmp_path / "x.csv"
    csv.touch()
    args = build_arg_parser().parse_args(["--csv", str(csv), "-p", "go"])

    @contextlib.asynccontextmanager
    async def fake_ctx(_path: str):
        yield MagicMock()

    with (
        patch("ada.cli.load_environment"),
        patch("ada.cli.AsyncSqliteSaver.from_conn_string", fake_ctx),
        patch("ada.cli.repl_loop", new=AsyncMock()) as repl,
        patch("ada.cli.run_agent_turn", new=AsyncMock()) as turn,
    ):
        await _amain(args)

    turn.assert_awaited_once()
    repl.assert_not_called()


async def test_amain_legacy_positional_one_shot_calls_run_agent_turn(
    tmp_path: pathlib.Path,
) -> None:
    """Positional <csv> <objective> runs a single turn and skips the REPL."""
    csv = tmp_path / "x.csv"
    csv.touch()
    args = build_arg_parser().parse_args([str(csv), "go"])

    @contextlib.asynccontextmanager
    async def fake_ctx(_path: str):
        yield MagicMock()

    with (
        patch("ada.cli.load_environment"),
        patch("ada.cli.AsyncSqliteSaver.from_conn_string", fake_ctx),
        patch("ada.cli.repl_loop", new=AsyncMock()) as repl,
        patch("ada.cli.run_agent_turn", new=AsyncMock()) as turn,
    ):
        await _amain(args)

    turn.assert_awaited_once()
    repl.assert_not_called()


async def test_amain_one_shot_requires_csv_when_prompt_given() -> None:
    """-p without --csv exits with an error before opening anything."""
    args = build_arg_parser().parse_args(["-p", "hi"])
    with (
        patch("ada.cli.load_environment"),
        pytest.raises(SystemExit),
    ):
        await _amain(args)


async def test_amain_csv_without_prompt_errors() -> None:
    """--csv without -p exits because the flag pair only describes one-shot mode."""
    args = build_arg_parser().parse_args(["--csv", "data.csv"])
    with (
        patch("ada.cli.load_environment"),
        pytest.raises(SystemExit),
    ):
        await _amain(args)


def test_parse_command_malformed_quoting_raises_slash_error() -> None:
    """Unterminated quoting in a slash command yields a user-facing SlashError."""
    with pytest.raises(SlashError, match="malformed"):
        parse_command('/csv "unterminated')


def test_safe_stem_passthrough_for_valid_characters() -> None:
    """Names already matching the safe set are returned unchanged."""
    assert _safe_stem("Titanic-Dataset") == "Titanic-Dataset"
    assert _safe_stem("housing_data.v2") == "housing_data.v2"


def test_safe_stem_replaces_unsafe_characters_with_underscore() -> None:
    """Spaces and other unsafe characters collapse to underscores."""
    assert _safe_stem("my data") == "my_data"
    assert _safe_stem("weird/name?") == "weird_name_"


def test_safe_stem_rejects_empty_after_sanitization() -> None:
    """A stem reduced to the empty string is rejected."""
    with pytest.raises(ValueError, match="empty"):
        _safe_stem("")


async def test_run_agent_turn_sanitizes_stem_with_spaces(tmp_path: pathlib.Path) -> None:
    """CSV filenames with spaces are passed to provision_workspace as a safe stem."""
    csv = tmp_path / "my data.csv"
    csv.write_text("a,b\n1,2\n")
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="thr-1",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )

    async def empty_stream(_inputs, _config=None, **_kwargs):  # noqa: ANN001
        return
        yield  # pragma: no cover

    fake_graph = MagicMock()
    fake_graph.astream = empty_stream
    backend = MagicMock()
    terminate = AsyncMock()
    from ada.runtime.workspace import SandboxResources

    resources = SandboxResources(backend=backend, terminate=terminate)

    with (
        patch(
            "ada.cli.provision_workspace",
            new=AsyncMock(return_value=(resources, tmp_path / "m")),
        ) as provision,
        patch("ada.cli.create_analytics_agent", return_value=fake_graph),
    ):
        await run_agent_turn(session, "go")

    provision.assert_awaited_once_with("my_data", "thr-1", csv.resolve())


async def test_run_agent_turn_catches_provisioning_errors(tmp_path: pathlib.Path) -> None:
    """A provision_workspace failure is caught; no sandbox to terminate."""
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="thr-1",
        checkpointer=MagicMock(),
        console=console,
    )
    with patch(
        "ada.cli.provision_workspace",
        new=AsyncMock(side_effect=RuntimeError("provisioning failed")),
    ):
        await run_agent_turn(session, "go")
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "Agent error: provisioning failed" in rendered


async def test_run_agent_turn_terminates_sandbox_when_graph_creation_fails(
    tmp_path: pathlib.Path,
) -> None:
    """If graph construction throws after provisioning, the sandbox is still terminated."""
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="thr-1",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )
    backend = MagicMock()
    terminate = AsyncMock()
    from ada.runtime.workspace import SandboxResources

    resources = SandboxResources(backend=backend, terminate=terminate)

    with (
        patch(
            "ada.cli.provision_workspace",
            new=AsyncMock(return_value=(resources, tmp_path / "m")),
        ),
        patch("ada.cli.create_analytics_agent", side_effect=RuntimeError("build failed")),
    ):
        await run_agent_turn(session, "go")

    terminate.assert_awaited_once()


async def test_run_agent_turn_terminates_sandbox_when_astream_fails(
    tmp_path: pathlib.Path,
) -> None:
    """If astream throws after the middleware was attached, terminate is still invoked."""
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="thr-1",
        checkpointer=MagicMock(),
        console=MagicMock(),
    )

    async def boom(_inputs, _config=None, **_kwargs):  # noqa: ANN001
        raise RuntimeError("boom")
        yield  # pragma: no cover

    fake_graph = MagicMock()
    fake_graph.astream = boom
    backend = MagicMock()
    terminate = AsyncMock()
    from ada.runtime.workspace import SandboxResources

    resources = SandboxResources(backend=backend, terminate=terminate)

    with (
        patch(
            "ada.cli.provision_workspace",
            new=AsyncMock(return_value=(resources, tmp_path / "m")),
        ),
        patch("ada.cli.create_analytics_agent", return_value=fake_graph),
    ):
        await run_agent_turn(session, "go")

    terminate.assert_awaited_once()


async def test_run_agent_turn_reraises_unexpected_exceptions_after_cleanup(
    tmp_path: pathlib.Path,
) -> None:
    """Unexpected exceptions are logged, cleaned up, and re-raised."""
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="thr-1",
        checkpointer=MagicMock(),
        console=console,
    )

    async def boom(_inputs, _config=None, **_kwargs):  # noqa: ANN001
        raise LookupError("surprise")
        yield  # pragma: no cover

    fake_graph = MagicMock()
    fake_graph.astream = boom
    backend = MagicMock()
    terminate = AsyncMock()
    from ada.runtime.workspace import SandboxResources

    resources = SandboxResources(backend=backend, terminate=terminate)

    with (
        patch(
            "ada.cli.provision_workspace",
            new=AsyncMock(return_value=(resources, tmp_path / "m")),
        ),
        patch("ada.cli.create_analytics_agent", return_value=fake_graph),
        pytest.raises(LookupError, match="surprise"),
    ):
        await run_agent_turn(session, "go")

    terminate.assert_awaited_once()
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "surprise" in rendered or "error" in rendered.lower()


async def test_repl_loop_second_ctrl_c_exits(tmp_path: pathlib.Path) -> None:
    """Two consecutive Ctrl-C presses at the prompt exit the loop."""
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=[KeyboardInterrupt, KeyboardInterrupt])
    with patch("ada.cli.PromptSession", return_value=fake_session):
        await repl_loop(session)


async def test_repl_loop_ctrl_c_counter_resets_after_input(tmp_path: pathlib.Path) -> None:
    """A successful input between Ctrl-C presses resets the exit counter."""
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(
        side_effect=[KeyboardInterrupt, "/help", KeyboardInterrupt, EOFError]
    )
    with patch("ada.cli.PromptSession", return_value=fake_session):
        await repl_loop(session)


async def test_repl_loop_handles_cancelled_turn(tmp_path: pathlib.Path) -> None:
    """A cancelled in-flight turn returns to the prompt instead of killing the REPL."""
    csv = tmp_path / "d.csv"
    csv.touch()
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=csv,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=["do it", EOFError])
    with (
        patch("ada.cli.PromptSession", return_value=fake_session),
        patch("ada.cli.run_agent_turn", new=AsyncMock(side_effect=asyncio.CancelledError)),
    ):
        await repl_loop(session)
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "cancel" in rendered.lower()


async def test_repl_loop_surfaces_malformed_slash_input(tmp_path: pathlib.Path) -> None:
    """Malformed quoting in a slash command is surfaced; the loop continues."""
    console = MagicMock()
    session = Session(
        cwd=tmp_path,
        csv_path=None,
        thread_id="t",
        checkpointer=MagicMock(),
        console=console,
    )
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=['/csv "unterminated', "/exit"])
    with patch("ada.cli.PromptSession", return_value=fake_session):
        await repl_loop(session)
    rendered = " ".join(repr(c.args[0]) for c in console.print.call_args_list)
    assert "malformed" in rendered.lower()
