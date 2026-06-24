"""``ada`` - interactive analytics REPL and one-shot CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.table import Table

from ada.agent import create_analytics_agent
from ada.config import load_environment
from ada.runtime.workspace import provision_workspace

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass
class Session:
    """Per-process REPL state.

    Attributes:
        cwd: Directory the user launched ``ada`` in. Used by ``/list``.
        csv_path: Currently selected CSV, or ``None`` until selected.
        thread_id: Active LangGraph thread and sandbox mirror id.
        checkpointer: Process-wide checkpointer shared across every turn.
        console: Rich console used for all CLI output.
    """

    cwd: pathlib.Path
    csv_path: pathlib.Path | None
    thread_id: str
    checkpointer: BaseCheckpointSaver[str]
    console: Console

    @property
    def stem(self) -> str | None:
        """Return the filename stem of the current CSV, or ``None``."""
        return self.csv_path.stem if self.csv_path is not None else None


class ExitRepl(Exception):
    """Raised by handlers such as ``/exit`` to terminate the REPL loop."""


class SlashError(Exception):
    """Raised by slash handlers for user-facing errors."""


def discover_csvs(directory: pathlib.Path) -> list[pathlib.Path]:
    """Return top-level ``*.csv`` files in ``directory``, sorted by name.

    Args:
        directory: Directory to scan. Does not recurse.

    Returns:
        Sorted list of CSV paths. Empty list if none are found.
    """
    return sorted(p for p in directory.glob("*.csv") if p.is_file())


def choose_csv(csvs: list[pathlib.Path], *, console: Console) -> pathlib.Path | None:
    """Pick a CSV from the supplied list, prompting the user if needed.

    Args:
        csvs: Candidate CSV paths. Sort order is preserved in the picker.
        console: Rich console for prompt rendering.

    Returns:
        Selected path, the only entry if there is exactly one, or ``None``.
    """
    if not csvs:
        return None
    if len(csvs) == 1:
        return csvs[0]

    console.print("[bold]Multiple CSVs found:[/bold]")
    for i, p in enumerate(csvs, start=1):
        console.print(f"  [cyan]{i}[/cyan]) {p.name}")

    while True:
        raw = input(f"Select [1-{len(csvs)}]: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            console.print("[yellow]Please enter a number.[/yellow]")
            continue
        if 1 <= idx <= len(csvs):
            return csvs[idx - 1]
        console.print(f"[yellow]Out of range (1-{len(csvs)}).[/yellow]")


def parse_thread_id(dirname: str) -> str | None:
    """Extract ``thread_id`` from a ``<stem>_<thread_id>`` directory name.

    Args:
        dirname: Directory name, e.g. ``"Titanic_abc-123"``.

    Returns:
        The substring after the last underscore, or ``None`` if absent.
    """
    if "_" not in dirname:
        return None
    return dirname.rsplit("_", 1)[1]


def list_thread_dirs(workspace_root: pathlib.Path, *, stem: str) -> list[pathlib.Path]:
    """List per-thread mirror directories for a given CSV stem.

    Args:
        workspace_root: Host-side ``workspace/`` directory.
        stem: Dataset stem to filter on.

    Returns:
        Matching directories sorted by mtime, most recent first.
    """
    if not workspace_root.is_dir():
        return []
    matches = [p for p in workspace_root.glob(f"{stem}_*") if p.is_dir()]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches


def open_report(report_path: pathlib.Path, *, console: Console) -> None:
    """Open a report file in the user's OS default application.

    Args:
        report_path: Absolute path to the report file.
        console: Rich console for status messages.
    """
    if not report_path.exists():
        console.print(f"[yellow]Report not found at {report_path}[/yellow]")
        return
    if sys.platform == "darwin":
        opener = ["open", str(report_path)]
    elif sys.platform == "win32":
        try:
            os.startfile(str(report_path))  # type: ignore[attr-defined]
        except OSError as exc:
            console.print(f"[yellow]Could not open {report_path}: {exc}[/yellow]")
        return
    else:
        opener = ["xdg-open", str(report_path)]
    try:
        subprocess.run(opener, check=False)
    except FileNotFoundError:
        console.print(
            f"[yellow]Could not open {report_path}: '{opener[0]}' is not installed. "
            f"Open the file manually.[/yellow]"
        )


def workspace_root() -> pathlib.Path:
    """Return the host-side ``workspace/`` directory."""
    return pathlib.Path(__file__).resolve().parent.parent / "workspace"


_UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_stem(name: str) -> str:
    """Coerce a filename stem to the safe character set used by ``provision_workspace``.

    Args:
        name: Raw stem (e.g. ``"my data"``).

    Returns:
        Stem with every character outside ``[A-Za-z0-9._-]`` replaced by ``_``.

    Raises:
        ValueError: If the input is empty.
    """
    if not name:
        raise ValueError("stem must not be empty")
    return _UNSAFE_STEM_CHARS.sub("_", name)


def parse_command(line: str) -> tuple[str, list[str]] | None:
    """Parse a REPL line into a slash command tuple if applicable.

    Args:
        line: Raw input line from the user.

    Returns:
        ``(verb, args)`` for slash commands; ``None`` for plain text.

    Raises:
        SlashError: If the line starts with ``/`` but its quoting is malformed.
    """
    stripped = line.strip()
    if not stripped.startswith("/") or stripped == "/":
        return None
    try:
        tokens = shlex.split(stripped[1:])
    except ValueError as exc:
        raise SlashError(f"malformed command: {exc}") from exc
    if not tokens:
        return None
    return tokens[0], tokens[1:]


async def _cmd_help(args: list[str], session: Session) -> None:
    """Print the available slash commands.

    Args:
        args: Ignored command arguments.
        session: Active REPL session whose console receives the help table.
    """
    del args
    table = Table(title="ada commands", show_header=True, header_style="bold")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_row("/help", "Show this help")
    table.add_row("/exit", "Quit (also Ctrl-D)")
    table.add_row("/new", "Start a fresh thread for the current CSV")
    table.add_row("/csv <path>", "Switch to a different CSV (starts a new thread)")
    table.add_row("/list", "List CSV files in the current directory")
    table.add_row("/resume [thread_id]", "Resume a past thread (picker if no id)")
    table.add_row("/open report", "Open the current thread's report.md")
    session.console.print(table)


async def _cmd_exit(args: list[str], session: Session) -> None:
    """Terminate the REPL.

    Args:
        args: Ignored command arguments.
        session: Active REPL session being shut down.

    Raises:
        ExitRepl: Always raised to stop the REPL loop.
    """
    del args, session
    raise ExitRepl


async def _cmd_new(args: list[str], session: Session) -> None:
    """Start a fresh LangGraph thread for the current CSV."""
    del args
    session.thread_id = str(uuid.uuid4())
    session.console.print(f"[green]New thread:[/green] {session.thread_id}")


async def _cmd_csv(args: list[str], session: Session) -> None:
    """Switch to another CSV and rotate the thread id."""
    if len(args) != 1:
        raise SlashError("usage: /csv <path>")
    candidate = pathlib.Path(args[0]).expanduser()
    if not candidate.is_absolute():
        candidate = (session.cwd / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_file():
        raise SlashError(f"{candidate} is not a file")
    session.csv_path = candidate
    session.thread_id = str(uuid.uuid4())
    session.console.print(
        f"[green]CSV:[/green] {candidate} [dim](new thread {session.thread_id})[/dim]"
    )


async def _cmd_list(args: list[str], session: Session) -> None:
    """List CSV files in the launch directory."""
    del args
    csvs = discover_csvs(session.cwd)
    if not csvs:
        session.console.print(f"[yellow]No CSV files in {session.cwd}[/yellow]")
        return
    session.console.print(f"[bold]CSV files in {session.cwd}:[/bold]")
    for p in csvs:
        marker = " [cyan](current)[/cyan]" if p == session.csv_path else ""
        session.console.print(f"  {p.name}{marker}")


async def _cmd_resume(args: list[str], session: Session) -> None:
    """Resume an existing thread for the current CSV."""
    if session.csv_path is None:
        raise SlashError("no csv selected - use /csv <path> first")
    if len(args) > 1:
        raise SlashError("usage: /resume [thread_id]")
    threads = list_thread_dirs(workspace_root(), stem=session.stem or "")
    if not threads:
        raise SlashError(f"no threads found for {session.stem}")

    if args:
        target_id = args[0]
        matches = [t for t in threads if parse_thread_id(t.name) == target_id]
        if not matches:
            raise SlashError(f"no thread {target_id!r} for {session.stem}")
        chosen = matches[0]
    else:
        session.console.print(f"[bold]Past threads for {session.stem} (newest first):[/bold]")
        for i, t in enumerate(threads, start=1):
            session.console.print(f"  [cyan]{i}[/cyan]) {parse_thread_id(t.name)}")
        while True:
            raw = input(f"Select [1-{len(threads)}]: ").strip()
            try:
                idx = int(raw)
            except ValueError:
                session.console.print("[yellow]Please enter a number.[/yellow]")
                continue
            if 1 <= idx <= len(threads):
                chosen = threads[idx - 1]
                break
            session.console.print(f"[yellow]Out of range (1-{len(threads)}).[/yellow]")

    tid = parse_thread_id(chosen.name)
    if tid is None:
        raise SlashError(f"could not parse thread_id from {chosen.name}")
    session.thread_id = tid
    session.console.print(f"[green]Resumed thread:[/green] {tid}")


async def _cmd_open(args: list[str], session: Session) -> None:
    """Open the report for the current thread."""
    if len(args) != 1 or args[0] != "report":
        raise SlashError("usage: /open report")
    if session.csv_path is None or session.stem is None:
        raise SlashError("no csv selected")
    mirror = workspace_root() / f"{session.stem}_{session.thread_id}"
    open_report(mirror / "report.md", console=session.console)


_HANDLERS: dict[str, Callable[[list[str], Session], Awaitable[None]]] = {
    "help": _cmd_help,
    "exit": _cmd_exit,
    "new": _cmd_new,
    "csv": _cmd_csv,
    "list": _cmd_list,
    "resume": _cmd_resume,
    "open": _cmd_open,
}


async def dispatch_slash(verb: str, args: list[str], session: Session) -> None:
    """Route a parsed slash command to its handler.

    Args:
        verb: Command name without the leading slash.
        args: Already-tokenized arguments.
        session: REPL session.
    """
    handler = _HANDLERS.get(verb)
    if handler is None:
        raise SlashError(f"unknown command: /{verb} - try /help")
    await handler(args, session)


async def run_agent_turn(session: Session, user_text: str) -> None:
    """Run a single agent turn end-to-end.

    Catches errors from provisioning, graph construction, or streaming and prints
    them to the console so the REPL stays alive. If a sandbox was provisioned
    but the turn fails before its lifecycle middleware can release it, the backend
    is terminated explicitly here.

    Args:
        session: Current REPL session.
        user_text: User's natural-language message.
    """
    if session.csv_path is None:
        session.console.print(
            "[yellow]No CSV selected. Use /csv <path> or /list to choose one.[/yellow]"
        )
        return

    csv_abs = session.csv_path.resolve()
    safe_stem = _safe_stem(csv_abs.stem)
    sandbox_resources = None
    try:
        sandbox_resources, mirror_root = await provision_workspace(
            safe_stem, session.thread_id, csv_abs
        )
        agent = create_analytics_agent(
            sandbox_resources.backend,
            mirror_root=mirror_root,
            terminate_sandbox=sandbox_resources.terminate,
            checkpointer=session.checkpointer,
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": session.thread_id,
                "csv_path": str(csv_abs),
                "stem": safe_stem,
                "__is_for_execution__": True,
            }
        }
        async for chunk in agent.astream(
            {"messages": [("user", user_text)]},
            config=config,
            stream_mode="updates",
            subgraphs=True,
            version="v2",
        ):
            if chunk["type"] == "updates":
                if "model" in chunk["data"]:
                    msg = chunk["data"]["model"]["messages"][-1]
                    if msg.content:
                        if isinstance(msg.content, list):
                            for content in msg.content:
                                if content["type"] == "thinking":
                                    session.console.print(
                                        f"[dim]{msg.name or 'agent'}(thinking): [/dim]",
                                        f"[dim]{content['thinking']}[/dim]",
                                    )
                                else:
                                    session.console.print(
                                        f"[bold green]🤖 {msg.name or 'agent'}:[/bold green] ",
                                        f"{content['text']}",
                                    )
                        else:
                            session.console.print(
                                f"[bold green]🤖 {msg.name or 'agent'}:[/bold green] {msg.content}"
                            )

                        session.console.print("=" * 50)

                elif "tools" in chunk["data"]:
                    msg = chunk["data"]["tools"]["messages"][-1]
                    if msg.content:
                        preview = msg.content[:150]
                        suffix = "..." if len(msg.content) > 150 else ""
                        session.console.print(
                            f"[italic cyan]🔧 {msg.name or 'tools'}:[/italic cyan] ",
                            f"{preview}{suffix}",
                        )

                        session.console.print("=" * 50)

    except asyncio.CancelledError:
        if sandbox_resources is not None:
            with contextlib.suppress(Exception):
                await sandbox_resources.terminate()
        raise
    except (RuntimeError, ValueError) as exc:
        session.console.print(f"[red]Agent error: {exc}[/red]")
        if sandbox_resources is not None:
            with contextlib.suppress(Exception):
                await sandbox_resources.terminate()
    except Exception as exc:  # noqa: BLE001
        session.console.print(f"[red]Agent error: {exc}[/red]")
        if sandbox_resources is not None:
            with contextlib.suppress(Exception):
                await sandbox_resources.terminate()
        raise


async def repl_loop(session: Session) -> None:
    """Run the interactive REPL until the user exits.

    Behavior:
        * Ctrl-D at the prompt exits cleanly.
        * One Ctrl-C at the prompt prints a hint; a second consecutive Ctrl-C exits.
        * Ctrl-C during an in-flight agent turn cancels the turn and returns to
          the prompt (Unix only — SIGINT handler is installed for the duration of
          the turn). On platforms without ``loop.add_signal_handler`` support the
          REPL falls back to catching ``KeyboardInterrupt`` from ``await``.

    Args:
        session: Mutable REPL state.
    """
    prompt = PromptSession()
    session.console.print("[dim]Type a question, or /help for commands. Ctrl-D to exit.[/dim]")
    consecutive_ctrl_c = 0
    while True:
        try:
            line = await prompt.prompt_async("> ")
        except EOFError:
            session.console.print("")
            return
        except KeyboardInterrupt:
            consecutive_ctrl_c += 1
            if consecutive_ctrl_c >= 2:
                session.console.print("[dim]Exiting.[/dim]")
                return
            session.console.print("[dim](press Ctrl-C again to exit, or use /exit / Ctrl-D)[/dim]")
            continue
        consecutive_ctrl_c = 0

        stripped = line.strip()
        if not stripped:
            continue

        try:
            parsed = parse_command(stripped)
        except SlashError as err:
            session.console.print(f"[red]{err}[/red]")
            continue
        if parsed is not None:
            verb, args = parsed
            try:
                await dispatch_slash(verb, args, session)
            except ExitRepl:
                return
            except SlashError as err:
                session.console.print(f"[red]{err}[/red]")
            continue

        await _run_turn_with_cancellation(session, stripped)


async def _run_turn_with_cancellation(session: Session, user_text: str) -> None:
    """Run an agent turn that can be cancelled by SIGINT (Ctrl-C) without exiting the REPL.

    Installs a temporary SIGINT handler that cancels the inner task; restores the
    previous handler when the turn completes. Falls back to ``KeyboardInterrupt``
    handling on platforms where the loop has no signal-handler support.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(run_agent_turn(session, user_text))
    handler_installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, task.cancel)
        handler_installed = True
    except (NotImplementedError, RuntimeError):
        # Windows / unsupported loop: signal-based cancellation isn't available.
        pass

    try:
        try:
            await task
        except (asyncio.CancelledError, KeyboardInterrupt):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            session.console.print("[yellow](turn cancelled)[/yellow]")
    finally:
        if handler_installed:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(signal.SIGINT)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``ada``.

    Returns:
        Parser with optional ``--csv`` and ``-p`` for one-shot mode.
    """
    parser = argparse.ArgumentParser(
        prog="ada",
        description=(
            "Interactive multi-subagent EDA orchestrator. Run with no arguments "
            "for a REPL in the current directory, or pass --csv and -p for one shot."
        ),
    )
    parser.add_argument("--csv", help="Path to a CSV file (enables one-shot mode).")
    parser.add_argument(
        "-p",
        "--prompt",
        help="Single objective to run non-interactively (requires --csv).",
    )
    parser.add_argument(
        "csv_pos",
        nargs="?",
        help="Path to a CSV file (legacy positional one-shot mode).",
    )
    parser.add_argument(
        "prompt_pos",
        nargs="?",
        help="Single objective to run non-interactively (legacy positional one-shot mode).",
    )
    return parser


async def _amain(args: argparse.Namespace) -> None:
    """Async entry point that sets up environment, checkpointing, and mode."""
    load_environment()

    cwd = Path.cwd()
    csv_path = args.csv or args.csv_pos
    prompt = args.prompt or args.prompt_pos

    one_shot = prompt is not None
    if one_shot and not csv_path:
        raise SystemExit("error: --csv or a CSV path is required when using a prompt/objective")
    if csv_path and not one_shot:
        raise SystemExit(
            "error: csv path requires a prompt/objective (one-shot mode); omit both for the REPL"
        )

    workspace = workspace_root()
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / ".checkpoints.sqlite"
    console = Console()

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        if one_shot:
            csv = Path(csv_path).expanduser().resolve()
            if not csv.is_file():
                raise SystemExit(f"error: {csv} is not a file")
            session = Session(
                cwd=cwd,
                csv_path=csv,
                thread_id=str(uuid.uuid4()),
                checkpointer=checkpointer,
                console=console,
            )
            console.print(
                f"[bold blue]ada (one-shot)[/bold blue] - {csv.name} - thread {session.thread_id}"
            )
            await run_agent_turn(session, prompt)
            return

        csvs = discover_csvs(cwd)
        chosen = choose_csv(csvs, console=console)
        if chosen is None and not csvs:
            console.print(f"[yellow]No CSV files in {cwd}. Use /csv <path> to choose one.[/yellow]")

        session = Session(
            cwd=cwd,
            csv_path=chosen,
            thread_id=str(uuid.uuid4()),
            checkpointer=checkpointer,
            console=console,
        )
        header_csv = chosen.name if chosen else "(none)"
        console.print(
            f"[bold blue]ada[/bold blue] - CSV: {header_csv} - thread {session.thread_id}"
        )
        await repl_loop(session)


def main() -> None:
    """Sync entry point registered as the ``ada`` console script."""
    args = build_arg_parser().parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
