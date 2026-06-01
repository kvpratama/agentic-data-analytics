# Agentic Data Analytics

A multi-subagent **Exploratory Data Analysis (EDA)** workflow powered by [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview). A single orchestrator coordinates three specialized subagents — **profiler**, **cleaner**, and **analyst** — that execute inside a secure, sandboxed Modal microVM to profile a CSV dataset, fix data quality issues, and produce an insights report with visualizations.

## What This Example Demonstrates

- **`SubAgentMiddleware`** — three named subagents with distinct system prompts and skills, communicating through the orchestrator via the `task` tool.
- **Modal Sandbox Isolation** — runs in a secure, isolated microVM using `ModalSandbox` (from `langchain-modal`). The agent cannot escape the container, keeping your host machine protected.
- **Single `execute` tool surface** — each subagent has the full power of pandas / scipy / matplotlib via one `execute` tool to run commands inside the sandbox container.
- **Skills (progressive disclosure)** — methodology lives in `SKILL.md` files under `skills/`, served through a host filesystem route and loaded on demand.
- **File-based state handoff** — subagents coordinate within a turn through explicit sandbox files (`/workspace/dataset.csv`, `/workspace/profile.json`, etc.) rather than in-memory kernel variables.
- **Thread-scoped follow-ups** — LangGraph Studio/API threads keep chat history, while host-side mirrors under `workspace/<dataset-stem>_<thread_id>/` preserve `/workspace/` artifacts across turns.
- **Multi-provider model configuration** — swap between Anthropic, OpenAI, or Google with a single `.env` change.

## Architecture

```text
                    ╭──────────────────╮
                    │   Orchestrator   │  (main deep agent)
                    ╰────────┬─────────╯
                             │ delegates via task()
            ╭────────────────┼────────────────╮
            ▼                ▼                ▼
      ╭──────────╮     ╭──────────╮     ╭──────────╮
      │ profiler │     │ cleaner  │     │ analyst  │
      ╰────┬─────╯     ╰────┬─────╯     ╰────┬─────╯
           │                │                │
           │ execute        │ execute        │ execute
           ╰────────────────┼────────────────╯
                            ▼
              ╭──────────────────────────╮
              │     CompositeBackend     │
              │ default → ModalSandbox   │
              │ /skills → Filesystem     │
              ╰──────┬──────────────┬────╯
                     │ default      │ /skills
                     ▼              ▼
    ╭──────────────────────────╮  ╭─────────────────────────╮
    │    ModalSandbox          │  │ FilesystemBackend       │
    │ /workspace/dataset.csv   │  │ local repo skills/      │
    │ /workspace/profile.json  │  │ read-only route         │
    ╰─────────┬────────────────╯  ╰─────────────────────────╯
              │ seeded/downloaded each turn
              ▼
        ╭───────────────────────────────╮
        │ workspace/<stem>_<thread_id>/ │  ← thread-scoped persistence
        ╰───────────────────────────────╯
```

The `CompositeBackend` routes ordinary sandbox execution and `/workspace/` file I/O to `ModalSandbox`, while `/skills/` reads are served from the local repository through `FilesystemBackend`. The `workspace/<stem>_<thread_id>/` mirror is handled by the sandbox seeding and lifecycle middleware, not by the `/skills/` backend route. A write-deny `FilesystemPermission` protects `/skills/**`, so agents can load skills but cannot modify them.

1. **Profiler** — loads the [profiling skill](skills/profiler_skills/profiler/SKILL.md), inspects `/workspace/dataset.csv` in the sandbox, and writes `/workspace/profile.json` with raw stats and a `diagnosis` list.
2. **Cleaner** — loads the [cleaning skill](skills/cleaner_skills/cleaner/SKILL.md), reads `/workspace/profile.json`, and applies cleaning steps (fill nulls, cast dtypes, clip outliers, drop duplicates, etc.) by writing the cleaned output to `/workspace/dataset.clean.csv`, leaving the original `/workspace/dataset.csv` unchanged.
3. **Analyst** — loads the [analysis skill](skills/analyst_skills/analyst/SKILL.md), reads `/workspace/dataset.clean.csv` (and optionally `/workspace/dataset.csv` for raw comparisons), runs correlations / aggregations / hypothesis tests tied to the user's objective, saves plots to `/workspace/plots/`, and writes the final `/workspace/report.md`.

## Quick Start

### Prerequisites

- Python 3.12 or higher
- An API key for your chosen model provider ([Anthropic](https://console.anthropic.com/), [OpenAI](https://platform.openai.com/), or [Google](https://aistudio.google.com/))
- A [Modal](https://modal.com/) account and authenticated credentials on your host machine (run `uv run modal token new` to log in, or set the standard `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` environment variables).

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

`ada` is the entrypoint. Run it in any directory containing CSV files for an
interactive REPL, or pass `--csv` and `-p` for a one-shot non-interactive turn.

### Interactive

```bash
uv run ada

# Set the CSV file
/csv ./dataset/housing.csv
```

If exactly one CSV is present it is auto-selected; otherwise you'll get a
numbered picker. After launch you're in a chat-style prompt:

```text
ada - CSV: Titanic-Dataset.csv - thread 8f4f12c...
Type a question, or /help for commands. Ctrl-D to exit.

> what columns are in this dataset?
profiler: profile.json written (12 columns, 891 rows)
assistant: The dataset has 12 columns: PassengerId, Survived, ...

> /open report
```

Slash commands:

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/exit` | Quit (also Ctrl-D) |
| `/new` | Start a fresh thread for the current CSV |
| `/csv <path>` | Switch CSV (starts a new thread) |
| `/list` | List CSVs in cwd |
| `/resume [thread_id]` | Resume a past thread (picker if no id) |
| `/open report` | Open the current thread's `report.md` |

Each turn provisions a fresh Modal sandbox, runs the agent, mirrors artifacts to
`workspace/<stem>_<thread_id>/`, and terminates the sandbox. Conversation
history persists across turns and across REPL sessions via
`workspace/.checkpoints.sqlite`.

### One-Shot

```bash
uv run ada --csv dataset/Titanic-Dataset.csv -p "Investigate factors that affected survival"
```

Runs a single turn and exits. The thread it creates is resumable from a later
interactive session with `/resume`.

Output artifacts land in a thread-scoped mirror such as `workspace/Titanic-Dataset_8f4f.../`:

```text
workspace/Titanic-Dataset_<thread_id>/
├── dataset.csv       ← raw input copied once for this thread
├── dataset.clean.csv ← cleaned copy (your original is untouched)
├── profile.json      ← profiler output
├── changes.json      ← cleaner log of changes
├── plots/*.png       ← analyst visualizations
└── report.md         ← final insights report
```

### LangGraph Studio

The repository includes `langgraph.json`, exposing the graph as `analytics`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "analytics": "./ada/agent.py:make_graph"
  },
  "env": ".env"
}
```

Run Studio from the project root:

```bash
uv run langgraph dev
```

Before submitting a run, open [**Manage Assistants**](https://docs.langchain.com/langsmith/use-studio#manage-assistants) in LangGraph Studio and set the active assistant's configurable payload:

```json
{
  "csv_path": "dataset/Titanic-Dataset.csv"
}
```

Studio supplies the run/thread metadata. Each turn creates a fresh Modal sandbox, seeds it from `workspace/<stem>_<thread_id>/`, runs the agent, mirrors artifacts back to that directory, and terminates the sandbox. To analyze a different dataset, update the active assistant's `csv_path` and start a new thread. Old thread mirrors can be removed manually from `workspace/` when they are no longer needed.

## Try it on more datasets

A few public datasets to play with — download via Kaggle's public endpoint and run the agent against them.

### Iris — species similarity

```bash
curl -L -o dataset/iris.zip \
    https://www.kaggle.com/api/v1/datasets/download/uciml/iris
unzip dataset/iris.zip -d dataset/

uv run ada --csv dataset/Iris.csv \
    -p "Which two species are the most physically similar?"
```

### Titanic — survival drivers

```bash
curl -L -o dataset/titanic-dataset.zip \
    https://www.kaggle.com/api/v1/datasets/download/yasserh/titanic-dataset
unzip dataset/titanic-dataset.zip -d dataset/

uv run ada --csv dataset/Titanic-Dataset.csv \
    -p "Investigate factors that affected survival"
```

### California housing — price drivers

```bash
curl -L -o dataset/california-housing-prices.zip \
    https://www.kaggle.com/api/v1/datasets/download/camnugent/california-housing-prices
unzip dataset/california-housing-prices.zip -d dataset/

uv run ada --csv dataset/housing.csv \
    -p "What are the three strongest predictors of high home values?"
```

### Diamonds — feature interactions

```bash
curl -L -o dataset/diamonds.zip \
    https://www.kaggle.com/api/v1/datasets/download/shivam2503/diamonds
unzip dataset/diamonds.zip -d dataset/

uv run ada --csv dataset/diamonds.csv \
    -p "Which feature combinations yield the greatest improvement in predicting diamond prices over single-feature models?"
```

## Project Structure

```text
agentic-data-analytics/
├── ada/
│   ├── agent.py                      ← LangGraph factory and orchestrator
│   ├── subagents.py                  ← Subagent definitions (profiler, cleaner, analyst)
│   ├── cli.py                        ← CLI entrypoint
│   ├── agent_middleware.py           ← mirrors /workspace artifacts and terminates sandboxes
│   ├── config.py                     ← Settings + get_model() (multi-provider + Modal settings)
│   ├── config_test.py                ← unit tests for Settings
│   └── runtime/
│       ├── modal_runtime.py          ← sandbox build, seed, and download helpers
│       ├── modal_runtime_test.py     ← unit tests for sandbox runtime operations
│       └── workspace.py              ← workspace mirroring and sandbox provisioning logic
├── skills/
│   ├── profiler_skills/profiler/SKILL.md
│   ├── cleaner_skills/cleaner/SKILL.md
│   ├── analyst_skills/analyst/SKILL.md
│   └── orchestrator_skills/orchestrator/SKILL.md
├── dataset/                          ← gitignored, downloaded on demand
├── workspace/                        ← gitignored, host-side downloaded artifacts per-run
├── pyproject.toml
├── langgraph.json                    ← LangGraph Studio/API graph entrypoint
├── .env.example
├── .gitignore
└── README.md                         ← this file
```

## Configuration

`.env` selects the model and credentials:

```env
# Default
MODEL=anthropic:claude-sonnet-4-5-20250929
MODEL_SMALL=anthropic:claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=...

# Or
# MODEL=openai:gpt-4o
# MODEL_SMALL=openai:gpt-4o-mini
# OPENAI_API_KEY=...

# Or
# MODEL=google_genai:gemini-2.0-flash
# MODEL_SMALL=google_genai:gemini-1.5-flash
# GOOGLE_API_KEY=...

# Optional Modal App configuration overrides
# MODAL_APP_NAME=agentic-data-analytics
# MODAL_SANDBOX_TIMEOUT=1200

TEMPERATURE=0.0
```

Optional LangSmith tracing variables are also recognized (see `.env.example`).

### User-Global Config

`ada` launches from arbitrary directories, so it cannot rely on a `./.env` being
present. Set your API keys in either:

- **Your shell profile** (highest priority):

  ```bash
  # In ~/.zshrc, ~/.bashrc, etc.
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

- **A user-global config file** at `~/.config/ada/config.env`:

  ```bash
  mkdir -p ~/.config/ada
  cat > ~/.config/ada/config.env <<'EOF'
  MODEL=anthropic:claude-sonnet-4-5-20250929
  MODEL_SMALL=anthropic:claude-3-5-sonnet-20241022
  ANTHROPIC_API_KEY=sk-ant-...
  EOF
  ```

Resolution order is **OS environment > `./.env` in cwd >
`~/.config/ada/config.env`**, with `override=False` at each step. Existing keys
are never replaced.

Modal credentials (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`) are read by the
Modal SDK directly from `~/.modal.toml` (created by `modal token new`) or the
corresponding environment variables.

## Extending This Example

1. **Feature engineer subagent** — add a `feature_engineer` step downstream of `cleaner` with a skill covering one-hot encoding, scaling, binning, datetime decomposition, and train/test splits. Outputs `/workspace/features.parquet`.
2. **Human-in-the-loop approval** — wrap `execute` in `interrupt_on={...}` plus a `MemorySaver` checkpointer and a CLI approve/reject/edit prompt loop, so destructive commands require confirmation.
3. **Cleaning script export** — extend the cleaning skill to emit an auditable `cleaning_pipeline.py` reproducing the applied operations.
4. **Multi-dataset orchestration** — accept a directory of CSVs and process each in its own concurrent `ModalSandbox`.

## Resources

- [Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [LangChain](https://www.langchain.com/)
- [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)
