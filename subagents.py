from deepagents import SubAgent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
)

from config import Settings, get_model, get_model_small

WORK_RULES = (
    "All dataset files live under '/workspace/'. Use absolute paths: "
    "'/workspace/dataset.csv' (raw, immutable — never overwrite), "
    "'/workspace/dataset.clean.csv' (cleaner output), "
    "'/workspace/profile.json', '/workspace/changes.json', "
    "'/workspace/report.md', '/workspace/plots/'. Skills live under '/skills/'."
)


def get_subagents(settings: Settings) -> list[SubAgent]:
    """Build the profiler, cleaner, and analyst subagents.

    Args:
        settings: Application settings used to configure models and retries.

    Returns:
        The configured subagent definitions.
    """

    model = get_model(settings)
    model_small = get_model_small(settings)

    def _make_middleware() -> list[AgentMiddleware]:
        """Build the shared middleware stack for each subagent.

        Returns:
            The retry and fallback middleware chain.
        """
        return [
            ModelRetryMiddleware(
                max_retries=settings.retry_max_retries,
                backoff_factor=settings.retry_backoff_factor,
                initial_delay=settings.retry_initial_delay,
            ),
            ModelFallbackMiddleware(model_small),
        ]

    profiler: SubAgent = {
        "name": "profiler",
        "description": "Profiling agent",
        "system_prompt": (
            "You are a data profiler. Your sole job is to inspect and describe "
            "the dataset as-is — do not clean, transform, or analyse it. "
            "Load the 'profiler' skill for the full methodology and judgement "
            "guidelines, then inspect '/workspace/dataset.csv' and write "
            "'/workspace/profile.json'.\n\n" + WORK_RULES
        ),
        "model": model,
        "middleware": _make_middleware(),
        "skills": ["/skills/profiler_skills/"],
    }

    cleaner: SubAgent = {
        "name": "cleaner",
        "description": "Cleaning agent",
        "system_prompt": (
            "You are a data cleaner. Your sole job is to fix data quality issues "
            "identified in '/workspace/profile.json' — do not analyse, summarise, or "
            "draw conclusions about the data. Load the 'cleaner' skill for the "
            "full methodology and judgement guidelines. Read '/workspace/profile.json' "
            "first, then apply fixes by reading '/workspace/dataset.csv' (raw, never "
            "modify it) and writing the cleaned output to '/workspace/dataset.clean.csv', "
            "plus '/workspace/changes.json' logging every decision made.\n\n" + WORK_RULES
        ),
        "model": model,
        "middleware": _make_middleware(),
        "skills": ["/skills/cleaner_skills/"],
    }

    analyst: SubAgent = {
        "name": "analyst",
        "description": "Analyst agent",
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
        "model": model,
        "middleware": _make_middleware(),
        "skills": ["/skills/analyst_skills/"],
    }

    return [profiler, cleaner, analyst]
