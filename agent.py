"""Multi-subagent EDA orchestrator using Deep Agents and a Modal sandbox.

Three subagents — profiler, cleaner, and analyst — share a single ephemeral
Modal microVM (``ModalSandbox`` backend). Each subagent loads its own
SKILL.md (progressive disclosure) for methodology and pandas/scipy snippets.

Usage:
    python agent.py <csv_path> <objective>

Example:
    python agent.py dataset/Titanic-Dataset.csv "Investigate factors that affected survival"
"""

import asyncio
import os
import pathlib
import sys

import modal
from deepagents import SubAgent, create_deep_agent
from langchain_core.runnables import RunnableConfig
from langchain_modal import ModalSandbox
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.panel import Panel

from config import get_model, get_settings
from tools.modal_runtime import build_image, download_artifacts, seed_sandbox

console = Console()

WORK_RULES = (
    "All dataset files live under '/work/'. Use absolute paths: "
    "'/work/dataset.csv' (raw, immutable — never overwrite), "
    "'/work/dataset.clean.csv' (cleaner output), "
    "'/work/profile.json', '/work/changes.json', "
    "'/work/report.md', '/work/plots/'. Skills live under '/skills/'."
)


def create_analytics_agent(backend: ModalSandbox) -> CompiledStateGraph:
    """Build the Deep Agent orchestrator with profiler, cleaner, and analyst subagents.

    Args:
        backend: A live ``ModalSandbox`` backend shared across all subagents.
            All filesystem operations and the auto-injected ``execute`` tool
            route through this sandbox.

    Returns:
        A configured Deep Agent ready to invoke with a user objective.
    """
    model = get_model()

    profiler: SubAgent = {
        "name": "profiler",
        "description": "Profiling agent",
        "system_prompt": (
            "You are a data profiler. Your sole job is to inspect and describe "
            "the dataset as-is — do not clean, transform, or analyse it. "
            "Load the 'profiler' skill for the full methodology and judgement "
            "guidelines, then inspect '/work/dataset.csv' and write "
            "'/work/profile.json'.\n\n" + WORK_RULES
        ),
        "model": model,
        "skills": ["/skills/profiler_skills/"],
    }

    cleaner: SubAgent = {
        "name": "cleaner",
        "description": "Cleaning agent",
        "system_prompt": (
            "You are a data cleaner. Your sole job is to fix data quality issues "
            "identified in '/work/profile.json' — do not analyse, summarise, or "
            "draw conclusions about the data. Load the 'cleaner' skill for the "
            "full methodology and judgement guidelines. Read '/work/profile.json' "
            "first, then apply fixes by reading '/work/dataset.csv' (raw, never "
            "modify it) and writing the cleaned output to '/work/dataset.clean.csv', "
            "plus '/work/changes.json' logging every decision made.\n\n" + WORK_RULES
        ),
        "model": model,
        "skills": ["/skills/cleaner_skills/"],
    }

    analyst: SubAgent = {
        "name": "analyst",
        "description": "Analyst agent",
        "system_prompt": (
            "You are a data analyst. Your sole job is to analyse and interpret "
            "the cleaned data — do not re-clean or re-profile it. Load the "
            "'analyst' skill for the full methodology and report structure. "
            "Read '/work/dataset.clean.csv' and '/work/changes.json' (you may "
            "consult '/work/dataset.csv' for raw comparisons), then produce "
            "'/work/report.md' and save any plots to '/work/plots/'. If the "
            "orchestrator passed a specific user question, lead the report with "
            "a direct answer to it.\n\n" + WORK_RULES
        ),
        "model": model,
        "skills": ["/skills/analyst_skills/"],
    }

    return create_deep_agent(
        model=model,
        system_prompt="""\
You are the Data Analytics Orchestrator. You have an `execute` tool (runs shell
commands inside an isolated sandbox) and three subagents: profiler, cleaner,
and analyst.

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
    """CLI entrypoint: parse args, start sandbox, run the agent, mirror artifacts."""
    if len(sys.argv) < 3:
        console.print("[red]Usage: python agent.py <csv_path> <objective>[/red]")
        sys.exit(1)

    csv_path = sys.argv[1]
    objective = " ".join(sys.argv[2:])

    if not os.path.exists(csv_path):
        console.print(f"[red]Error: CSV file '{csv_path}' not found.[/red]")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    local_root = pathlib.Path(f"work/{stem}")

    console.print(
        Panel(
            f"[bold blue]Starting EDA for: {stem}[/bold blue]\n"
            f"[italic]Objective: {objective}[/italic]"
        )
    )

    settings = get_settings()
    app = modal.App.lookup(settings.modal_app_name, create_if_missing=True)
    modal_sandbox = modal.Sandbox.create(
        image=build_image(),
        app=app,
        timeout=settings.modal_sandbox_timeout,
        cpu=2.0,
        memory=4096,
    )
    try:
        backend = ModalSandbox(sandbox=modal_sandbox)
        seed_sandbox(backend, csv_path=csv_path, skills_dir="skills")

        agent = create_analytics_agent(backend)
        config = RunnableConfig({"configurable": {}})
        async for chunk in agent.astream({"messages": [("user", objective)]}, config=config):
            if "model" in chunk:
                msg = chunk["model"]["messages"][-1]
                if msg.content:
                    console.print(f"[dim]{msg.name or 'agent'}:[/dim] {msg.content}")
            elif "tools" in chunk:
                msg = chunk["tools"]["messages"][-1]
                if msg.content:
                    console.print(f"[italic]{msg.name or 'agent'}:[/italic] {msg.content}")

        download_artifacts(backend, local_root=local_root)

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
    finally:
        modal_sandbox.terminate()


if __name__ == "__main__":
    asyncio.run(main())
