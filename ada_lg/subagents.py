"""Subagent specifications for the pure-LangGraph orchestrator."""

from __future__ import annotations

from ada.config import Settings

WORK_RULES = (
    "All dataset files live under '/workspace/'. Use absolute paths: "
    "'/workspace/dataset.csv' (raw, immutable — never overwrite), "
    "'/workspace/dataset.clean.csv' (cleaner output), "
    "'/workspace/profile.json', '/workspace/changes.json', "
    "'/workspace/report.md', '/workspace/plots/'. Skills live under '/skills/'."
)


def get_subagent_specs(settings: Settings) -> list[dict]:
    """Return profiler, cleaner, and analyst specifications.

    Args:
        settings: Settings kept for signature parity with ada.subagents.

    Returns:
        Plain dict specs consumed by ada_lg.agent.
    """
    del settings
    return [
        {
            "name": "profiler",
            "system_prompt": (
                "You are a data profiler. Your sole job is to inspect and describe "
                "the dataset as-is — do not clean, transform, or analyse it. "
                "Load the 'profiler' skill for the full methodology and judgement "
                "guidelines, then inspect '/workspace/dataset.csv' and write "
                "'/workspace/profile.json'.\n\n" + WORK_RULES
            ),
            "skill_dir": "/skills/profiler_skills/",
        },
        {
            "name": "cleaner",
            "system_prompt": (
                "You are a data cleaner. Your sole job is to fix data quality issues "
                "identified in '/workspace/profile.json' — do not analyse, summarise, or "
                "draw conclusions about the data. Load the 'cleaner' skill for the "
                "full methodology and judgement guidelines. Read '/workspace/profile.json' "
                "first, then apply fixes by reading '/workspace/dataset.csv' (raw, never "
                "modify it) and writing the cleaned output to '/workspace/dataset.clean.csv', "
                "plus '/workspace/changes.json' logging every decision made.\n\n" + WORK_RULES
            ),
            "skill_dir": "/skills/cleaner_skills/",
        },
        {
            "name": "analyst",
            "system_prompt": (
                "You are a data analyst. Your sole job is to analyse and interpret "
                "the cleaned data — do not re-clean or re-profile it. Load the "
                "'analyst' skill for the full methodology and report structure. "
                "Read '/workspace/dataset.clean.csv' and '/workspace/changes.json' (you may "
                "consult '/workspace/dataset.csv' for raw comparisons), then produce "
                "'/workspace/report.md' and save any plots to '/workspace/plots/'. If the "
                "orchestrator passed a specific user question, lead the report with "
                "a direct answer to it.\n\n" + WORK_RULES
            ),
            "skill_dir": "/skills/analyst_skills/",
        },
    ]
