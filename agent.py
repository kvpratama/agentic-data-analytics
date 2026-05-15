"""Multi-subagent EDA orchestrator using Deep Agents.

Three subagents — profiler, cleaner, and analyst — share a single persistent
IPython kernel via an ``execute_python`` tool. Each subagent loads its own
SKILL.md (progressive disclosure) for methodology and pandas/scipy snippets.

Usage:
    python agent.py <csv_path> <objective>

Example:
    python agent.py dataset/Titanic-Dataset.csv "Investigate factors that affected survival"
"""

import asyncio
import os
import shutil
import sys

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from config import get_model
from tools.code_executor import KernelSession, make_execute_python_tool

load_dotenv()

console = Console()


def create_analytics_agent(root_dir: str, session: KernelSession):
    """Build the Deep Agent orchestrator with profiler, cleaner, and analyst subagents.

    Args:
        root_dir: Per-dataset sandbox directory containing ``dataset.csv``.
        session: Live ``KernelSession`` shared across all subagents.

    Returns:
        A configured Deep Agent ready to invoke with a user objective.
    """
    model = get_model()
    execute_python = make_execute_python_tool(session)

    path_rules = (
        "PATH RULES (critical): When using execute_python, ALWAYS use bare "
        "relative paths like 'dataset.csv', 'profile.json', 'plots/foo.png'. "
        "NEVER use a leading '/' (e.g. '/dataset.csv') in execute_python — the "
        "kernel runs in the working directory, so a leading '/' resolves to "
        "the real OS root and will fail with 'No such file'. The '/'-prefixed "
        "paths you see from ls/read_file/write_file are virtual filesystem "
        "paths and only apply to those tools, not to execute_python."
    )

    profiler: SubAgent = {
        "name": "profiler",
        "description": "Profiling agent",
        "system_prompt": (
            "You are a data profiler. Load the 'profiling' skill for "
            "methodology and snippets, then use execute_python to inspect "
            "'dataset.csv' and write 'profile.json'.\n\n" + path_rules
        ),
        "model": model,
        "tools": [execute_python],
        "skills": ["/skills/profiler_skills/"],
    }

    cleaner: SubAgent = {
        "name": "cleaner",
        "description": "Cleaning agent",
        "system_prompt": (
            "You are a data cleaner. Load the 'cleaning' skill for "
            "methodology and snippets. Read 'profile.json' first, then use "
            "execute_python to apply cleaning steps and overwrite "
            "'dataset.csv'.\n\n" + path_rules
        ),
        "model": model,
        "tools": [execute_python],
        "skills": ["/skills/cleaner_skills/"],
    }

    analyst: SubAgent = {
        "name": "analyst",
        "description": "Analyst agent",
        "system_prompt": (
            "You are a data analyst. Load the 'analysis' skill for "
            "methodology and snippets. Read the cleaned 'dataset.csv', run "
            "analyses tied to the user's objective, save plots to 'plots/', "
            "and write 'report.md'.\n\n" + path_rules
        ),
        "model": model,
        "tools": [execute_python],
        "skills": ["/skills/analyst_skills/"],
    }

    backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)

    return create_deep_agent(
        model=model,
        system_prompt="""\
You are the Data Analytics Orchestrator.
Your goal is to coordinate a multi-step EDA workflow:
1. Delegate to 'profiler' to understand the dataset.
2. Delegate to 'cleaner' to fix data quality issues using the profiler's diagnosis.
3. Delegate to 'analyst' to answer the user's objective and produce report.md.
Use the 'task' tool to communicate with subagents.
The final result should be a 'report.md' in the working directory.""",
        subagents=[profiler, cleaner, analyst],
        backend=backend,
    )


async def main() -> None:
    """CLI entrypoint: parse args, set up sandbox, run the agent."""
    if len(sys.argv) < 3:
        console.print("[red]Usage: python agent.py <csv_path> <objective>[/red]")
        sys.exit(1)

    csv_path = sys.argv[1]
    objective = sys.argv[2]

    if not os.path.exists(csv_path):
        console.print(f"[red]Error: CSV file '{csv_path}' not found.[/red]")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    work_dir = f"work/{stem}"

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy(csv_path, os.path.join(work_dir, "dataset.csv"))

    # Mirror project skills into the sandbox so SkillsMiddleware (which shares
    # the agent's virtual-mode FilesystemBackend) can discover them at the
    # virtual path "/skills/".
    skills_src = os.path.abspath("skills")
    if os.path.isdir(skills_src):
        shutil.copytree(skills_src, os.path.join(work_dir, "skills"))

    console.print(
        Panel(
            f"[bold blue]Starting EDA for: {stem}[/bold blue]\n"
            f"[italic]Objective: {objective}[/italic]"
        )
    )

    session = KernelSession(work_dir=work_dir, timeout=60)
    try:
        agent = create_analytics_agent(work_dir, session)
        config = {"configurable": {"work_dir": work_dir}}
        async for chunk in agent.astream({"messages": [("user", objective)]}, config=config):
            if "model" in chunk:
                msg = chunk["model"]["messages"][-1]
                if msg.content:
                    console.print(f"[dim]{msg.name or 'agent'}:[/dim] {msg.content}")
            elif "tools" in chunk:
                msg = chunk["tools"]["messages"][-1]
                if msg.content:
                    console.print(f"[italic]{msg.name or 'agent'}:[/italic] {msg.content}")

        report_path = os.path.join(work_dir, "report.md")
        if os.path.exists(report_path):
            console.print(
                Panel(
                    f"[bold green]Analysis complete![/bold green]\n"
                    f"Final report saved to: [underline]{report_path}[/underline]"
                )
            )
        else:
            console.print("[yellow]Warning: report.md was not generated.[/yellow]")
    finally:
        session.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
