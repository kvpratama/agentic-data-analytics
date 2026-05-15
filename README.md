# Agentic Data Analytics

A multi-subagent **Exploratory Data Analysis (EDA)** workflow powered by [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview). A single orchestrator coordinates three specialized subagents — **profiler**, **cleaner**, and **analyst** — that share a long-lived IPython kernel to profile a CSV dataset, fix data quality issues, and produce an insights report with visualizations.

## What This Example Demonstrates

- **`SubAgentMiddleware`** — three named subagents with distinct system prompts and skills, communicating through the orchestrator via the `task` tool.
- **Persistent IPython kernel** — every subagent shares one [`KernelSession`](tools/code_executor.py) so DataFrames, imports, and intermediate variables survive across tool calls and across subagents.
- **Single `execute_python` tool surface** — each subagent has the full power of pandas / scipy / matplotlib in one tool. Smaller tool schemas, fewer choice tokens, more expressive analyses.
- **Skills (progressive disclosure)** — methodology lives in `SKILL.md` files under `skills/`, mounted into the agent's virtual filesystem and loaded on demand. System prompts stay short.
- **Sandboxed filesystem** — `FilesystemBackend(virtual_mode=True)` restricts the agent to a per-dataset working directory, keeping mutations safely recoverable.
- **Multi-provider model configuration** — swap between Anthropic, OpenAI, or Google with a single `.env` change.

## Architecture

```
                    ╭──────────────────╮
                    │   Orchestrator   │  (main deep agent)
                    │  • write_todos   │
                    │  • task          │
                    │  • filesystem    │
                    ╰────────┬─────────╯
                             │ delegates via task()
            ╭────────────────┼────────────────╮
            ▼                ▼                ▼
      ╭──────────╮     ╭──────────╮     ╭──────────╮
      │ profiler │     │ cleaner  │     │ analyst  │
      ╰────┬─────╯     ╰────┬─────╯     ╰────┬─────╯
           │                │                │
           │ execute_python │ execute_python │ execute_python
           ╰────────────────┼────────────────╯
                            ▼
              ╭──────────────────────────╮
              │     KernelSession        │  ← one shared IPython kernel
              │  cwd = work/<dataset>    │     state persists across calls
              ╰──────────────────────────╯
```

1. **Profiler** — loads the [profiling skill](skills/profiler_skills/profiling/SKILL.md), inspects `dataset.csv` (dtypes, missingness, cardinality, distributions, outliers, duplicates), and writes `profile.json` with raw stats and a `diagnosis` list.
2. **Cleaner** — loads the [cleaning skill](skills/cleaner_skills/cleaning/SKILL.md), reads `profile.json`, and applies cleaning steps (fill nulls, cast dtypes, clip outliers, drop duplicates, etc.) by overwriting `dataset.csv`.
3. **Analyst** — loads the [analysis skill](skills/analyst_skills/analysis/SKILL.md), runs correlations / aggregations / hypothesis tests tied to the user's objective, saves plots to `plots/`, and writes the final `report.md`.

## Quick Start

### Prerequisites

- Python 3.12 or higher
- An API key for your chosen model provider ([Anthropic](https://console.anthropic.com/), [OpenAI](https://platform.openai.com/), or [Google](https://aistudio.google.com/))

### Installation

1. Clone the repository and navigate to this example:

   ```bash
   git clone https://github.com/kvpratama/agentic-data-analytics.git
   cd agentic-data-analytics
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   uv sync
   ```

3. Set up your environment variables:

   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

4. Fetch a dataset (Titanic by default):

   ```bash
   mkdir -p dataset
   curl -L -o dataset/titanic-dataset.zip \
       https://www.kaggle.com/api/v1/datasets/download/yasserh/titanic-dataset
   unzip dataset/titanic-dataset.zip -d dataset/
   ```

   Or drop any CSV of your own into `dataset/`.

## Usage

Run the agent with a CSV path and a natural-language objective:

```bash
uv run python agent.py dataset/Titanic-Dataset.csv "Investigate factors that affected survival"
```

The agent will:

1. Copy the CSV into a fresh sandboxed `work/Titanic-Dataset/` directory.
2. Profile the dataset and identify quality issues (`profile.json`).
3. Clean the data in place on the sandboxed `dataset.csv`.
4. Analyze the cleaned data, generating plots and a final report.
5. Write `report.md` summarizing findings.

Output artifacts land in `work/<dataset-stem>/`:

```
work/Titanic-Dataset/
├── dataset.csv       ← cleaned copy (your original is untouched)
├── profile.json      ← profiler output
├── skills/           ← copy of skills/ for SkillsMiddleware lookup
├── plots/*.png       ← analyst visualizations
└── report.md         ← final insights report
```

## Try it on more datasets

A few public datasets to play with — download via Kaggle's public endpoint and run the agent against them.

### Iris — species similarity

```bash
curl -L -o dataset/iris.zip \
    https://www.kaggle.com/api/v1/datasets/download/uciml/iris
unzip dataset/iris.zip -d dataset/

uv run python agent.py dataset/Iris.csv \
    "Which two species are the most physically similar?"
```

### Titanic — survival drivers

```bash
curl -L -o dataset/titanic-dataset.zip \
    https://www.kaggle.com/api/v1/datasets/download/yasserh/titanic-dataset
unzip dataset/titanic-dataset.zip -d dataset/

uv run python agent.py dataset/Titanic-Dataset.csv \
    "Investigate factors that affected survival"
```

### California housing — price drivers

```bash
curl -L -o dataset/california-housing-prices.zip \
    https://www.kaggle.com/api/v1/datasets/download/camnugent/california-housing-prices
unzip dataset/california-housing-prices.zip -d dataset/

uv run python agent.py dataset/housing.csv \
    "What are the three strongest predictors of high home values?"
```

### Diamonds — feature interactions

```bash
curl -L -o dataset/diamonds.zip \
    https://www.kaggle.com/api/v1/datasets/download/shivam2503/diamonds
unzip dataset/diamonds.zip -d dataset/

uv run python agent.py dataset/diamonds.csv \
    "Which feature combinations yield the greatest improvement in predicting diamond prices over single-feature models?"
```

## Project Structure

```
agentic-data-analytics/
├── agent.py                          ← orchestrator + CLI entrypoint
├── config.py                         ← Settings + get_model() (multi-provider)
├── tools/
│   ├── code_executor.py              ← KernelSession + execute_python tool
│   └── code_executor_test.py         ← unit tests for the kernel session
├── skills/
│   ├── profiler_skills/profiling/SKILL.md
│   ├── cleaner_skills/cleaning/SKILL.md
│   └── analyst_skills/analysis/SKILL.md
├── dataset/                          ← gitignored, downloaded on demand
├── work/                             ← gitignored, agent-managed per-run outputs
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md                         ← this file
```

## Configuration

`.env` selects the model and credentials:

```env
# Default
MODEL=anthropic:claude-sonnet-4-5-20250929
ANTHROPIC_API_KEY=...

# Or
# MODEL=openai:gpt-4o
# OPENAI_API_KEY=...

# Or
# MODEL=google_genai:gemini-2.0-flash
# GOOGLE_API_KEY=...

TEMPERATURE=0.0
```

Optional LangSmith tracing variables are also recognized (see `.env.example`).

## Extending This Example

1. **Feature engineer subagent** — add a `feature_engineer` step downstream of `cleaner` with a skill covering one-hot encoding, scaling, binning, datetime decomposition, and train/test splits. Outputs `features.parquet`.
2. **Human-in-the-loop approval** — wrap `execute_python` in `interrupt_on={...}` plus a `MemorySaver` checkpointer and a CLI approve/reject/edit prompt loop, so destructive cells require confirmation.
3. **Cleaning script export** — extend the cleaning skill to emit an auditable `cleaning_pipeline.py` reproducing the applied operations.
4. **Multi-dataset orchestration** — accept a directory of CSVs and process each in its own per-dataset sandbox (each with its own `KernelSession`).

## Resources

- [Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [LangChain](https://www.langchain.com/)
- [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)
