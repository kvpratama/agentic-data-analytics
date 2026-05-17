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
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.panel import Panel

from config import get_model
from tools.code_executor import KernelSession, make_execute_python_tool

console = Console()


def create_analytics_agent(root_dir: str, session: KernelSession) -> CompiledStateGraph:
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
            "You are a data profiler. Your sole job is to inspect and describe "
            "the dataset as-is — do not clean, transform, or analyse it. "
            "Load the 'profiler' skill for the full methodology and judgement "
            "guidelines, then use execute_python to inspect 'dataset.csv' and "
            "write 'profile.json'.\n\n" + path_rules
        ),
        "model": model,
        "tools": [execute_python],
        "skills": ["/skills/profiler_skills/"],
    }

    cleaner: SubAgent = {
        "name": "cleaner",
        "description": "Cleaning agent",
        "system_prompt": (
            "You are a data cleaner. Your sole job is to fix data quality issues "
            "identified in profile.json — do not analyse, summarise, or draw "
            "conclusions about the data. Load the 'cleaner' skill for the full "
            "methodology and judgement guidelines. Read 'profile.json' first, "
            "then use execute_python to apply fixes to 'dataset.csv' in place "
            "and write 'changes.json' logging every decision made.\n\n" + path_rules
        ),
        "model": model,
        "tools": [execute_python],
        "skills": ["/skills/cleaner_skills/"],
    }

    analyst: SubAgent = {
        "name": "analyst",
        "description": "Analyst agent",
        "system_prompt": (
            "You are a data analyst. Your sole job is to analyse and interpret "
            "the cleaned data — do not re-clean or re-profile it. Load the "
            "'analyst' skill for the full methodology and report structure. "
            "Read 'dataset.csv' and 'changes.json', then produce 'report.md' "
            "and save any plots to 'plots/'. If the orchestrator passed a "
            "specific user question, lead the report with a direct answer to "
            "it.\n\n" + path_rules
        ),
        "model": model,
        "tools": [execute_python],
        "skills": ["/skills/analyst_skills/"],
    }

    backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)

    return create_deep_agent(
        model=model,
        tools=[execute_python],
        system_prompt="""\
You are the Data Analytics Orchestrator. You have execute_python available directly
and three subagents: profiler, cleaner, and analyst.

Load the 'orchestrator' skill before deciding how to proceed. It contains your
decision framework, routing guidance, and examples.

Your goal is to satisfy the user's objective — which may be a specific question,
an instruction, or a full EDA request — using whichever combination of tools and
subagents is appropriate. The final deliverable is either a direct answer, a
report.md, or both.""",
        subagents=[profiler, cleaner, analyst],
        backend=backend,
        skills=["/skills/orchestrator_skills/"],
    )


async def main() -> None:
    """CLI entrypoint: parse args, set up sandbox, run the agent."""
    if len(sys.argv) < 3:
        console.print("[red]Usage: python agent.py <csv_path> <objective>[/red]")
        sys.exit(1)

    csv_path = sys.argv[1]
    objective = " ".join(sys.argv[2:])

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
        config = RunnableConfig({"configurable": {"work_dir": work_dir}})
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
