import asyncio
import os
import pathlib
import sys
import uuid

from langchain_core.runnables import RunnableConfig
from rich.console import Console
from rich.panel import Panel

from agent import make_graph
from runtime.workspace import get_mirror_root

console = Console()


async def main() -> None:
    """CLI entrypoint: parse args, build a graph, and stream one turn."""
    if len(sys.argv) < 3:
        console.print("[red]Usage: python -m cli <csv_path> <objective>[/red]")
        sys.exit(1)

    csv_path = sys.argv[1]
    objective = " ".join(sys.argv[2:])

    if not os.path.exists(csv_path):
        console.print(f"[red]Error: CSV file '{csv_path}' not found.[/red]")
        sys.exit(1)

    csv_abs = pathlib.Path(csv_path).resolve()
    stem = csv_abs.stem
    thread_id = str(uuid.uuid4())
    local_root = get_mirror_root(stem, thread_id)

    console.print(
        Panel(
            f"[bold blue]Starting EDA for: {stem}[/bold blue]\n"
            f"[italic]Objective: {objective}[/italic]"
        )
    )

    config: RunnableConfig = {
        "configurable": {
            "csv_path": str(csv_abs),
            "stem": stem,
            "thread_id": thread_id,
            "__is_for_execution__": True,
        }
    }
    agent = await make_graph(config)
    async for chunk in agent.astream({"messages": [("user", objective)]}, config=config):
        if "model" in chunk:
            msg = chunk["model"]["messages"][-1]
            if msg.content:
                console.print(f"[dim]{msg.name or 'agent'}:[/dim] {msg.content}")
        elif "tools" in chunk:
            msg = chunk["tools"]["messages"][-1]
            if msg.content:
                console.print(f"[italic]{msg.name or 'agent'}:[/italic] {msg.content}")

    report_path = local_root / "report.md"
    if report_path.exists():
        console.print(
            Panel(
                f"[bold green]Analysis complete![/bold green]\n"
                f"Final report saved to: [underline]{report_path}[/underline]"
            )
        )
    else:
        console.print("[yellow]Warning: report.md was not generated.[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
