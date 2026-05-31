"""``ada`` - interactive analytics REPL and one-shot CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import pathlib
import shlex
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

from agent import create_analytics_agent
from config import load_environment
from runtime.workspace import provision_workspace

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
        subprocess.run(["open", str(report_path)], check=False)
    elif sys.platform == "win32":
        os.startfile(str(report_path))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(report_path)], check=False)


def workspace_root() -> pathlib.Path:
    """Return the host-side ``workspace/`` directory."""
    return pathlib.Path(__file__).resolve().parent / "workspace"


def parse_command(line: str) -> tuple[str, list[str]] | None:
    """Parse a REPL line into a slash command tuple if applicable.

    Args:
        line: Raw input line from the user.

    Returns:
        ``(verb, args)`` for slash commands; ``None`` for plain text.
    """
    stripped = line.strip()
    if not stripped.startswith("/") or stripped == "/":
        return None
    tokens = shlex.split(stripped[1:])
    if not tokens:
        return None
    return tokens[0], tokens[1:]


async def _cmd_help(args: list[str], session: Session) -> None:
    """Print command help."""
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
    """Terminate the REPL."""
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
    sandbox_resources, mirror_root = await provision_workspace(
        csv_abs.stem, session.thread_id, csv_abs
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
            "stem": csv_abs.stem,
            "__is_for_execution__": True,
        }
    }

    try:
        async for chunk in agent.astream({"messages": [("user", user_text)]}, config=config):
            if "model" in chunk:
                msg = chunk["model"]["messages"][-1]
                if msg.content:
                    session.console.print(f"[dim]{msg.name or 'agent'}:[/dim] {msg.content}")
            elif "tools" in chunk:
                msg = chunk["tools"]["messages"][-1]
                if msg.content:
                    session.console.print(f"[italic]{msg.name or 'agent'}:[/italic] {msg.content}")
    except Exception as exc:  # noqa: BLE001
        session.console.print(f"[red]Agent error: {exc}[/red]")


async def repl_loop(session: Session) -> None:
    """Run the interactive REPL until the user exits.

    Args:
        session: Mutable REPL state.
    """
    prompt = PromptSession()
    session.console.print("[dim]Type a question, or /help for commands. Ctrl-D to exit.[/dim]")
    while True:
        try:
            line = await prompt.prompt_async("> ")
        except EOFError:
            session.console.print("")
            return
        except KeyboardInterrupt:
            session.console.print(
                "[dim](use /exit or Ctrl-D to quit; Ctrl-C interrupts in-flight turns)[/dim]"
            )
            continue

        stripped = line.strip()
        if not stripped:
            continue

        parsed = parse_command(stripped)
        if parsed is not None:
            verb, args = parsed
            try:
                await dispatch_slash(verb, args, session)
            except ExitRepl:
                return
            except SlashError as err:
                session.console.print(f"[red]{err}[/red]")
            continue

        task = asyncio.create_task(run_agent_turn(session, stripped))
        try:
            await task
        except KeyboardInterrupt:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            session.console.print("[yellow](turn cancelled)[/yellow]")


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
    return parser


async def _amain(args: argparse.Namespace) -> None:
    """Async entry point that sets up environment, checkpointing, and mode."""
    load_environment()

    cwd = Path.cwd()
    one_shot = args.prompt is not None
    if one_shot and not args.csv:
        raise SystemExit("error: --csv is required when using -p")

    workspace = workspace_root()
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / ".checkpoints.sqlite"
    console = Console()

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        if one_shot:
            csv = Path(args.csv).expanduser().resolve()
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
            await run_agent_turn(session, args.prompt)
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
