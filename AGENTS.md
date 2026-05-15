## Project Overview

Multi-subagent EDA orchestrator using Deep Agents and a persistent IPython kernel. The system coordinates three specialized subagents to perform end-to-end data analysis:
1. **Profiler**: Understands the dataset structure, types, and summary statistics.
2. **Cleaner**: Fixes data quality issues based on the profile (handling missing values, types, etc.).
3. **Analyst**: Performs deep-dive analysis, generates visualizations, and writes the final report.

All subagents share a single IPython kernel, allowing them to build on each other's code and state.

---

## Tech Stack

- **Runtime**: Python 3.12+
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) & [Deep Agents](https://github.com/google-deepmind/deepagents)
- **Execution**: `ipykernel` and `jupyter-client` for persistent code execution
- **Data Science**: `pandas`, `scipy`, `matplotlib`, `seaborn`
- **Models**: Anthropic (default), OpenAI, or Google GenAI via LangChain `init_chat_model`
- **Environment**: `uv` for dependency management, `pydantic-settings` for configuration

---

## Project Structure

- `agent.py`: Orchestrator logic and subagent definitions.
- `config.py`: Centralized configuration and model initialization.
- `tools/`: Core tools, including the persistent `code_executor`.
- `skills/`: Domain-specific `SKILL.md` files for profiler, cleaner, and analyst roles.
- `work/`: Ephemeral sandbox directories for each analysis run.
- `dataset/`: Sample CSV files for testing.
- `pyproject.toml`: Dependency and tool configuration (Ruff, Pytest).

---

## Common Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run ty check

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>
```

---

## Code Conventions

### General
- Python **3.12+** minimum.
- **Async-first where applicable** — use `async def` for all FastAPI route handlers and any I/O-bound functions (DB calls, LLM calls).
- **Strict type hints** on every function signature, including return types. No bare `Any` unless unavoidable.
- **Docstrings on every function and class** using Google-style format.

### Environment Variables
- All secrets and configuration come from `.env` via `python-dotenv`.
- Access config only through the `Settings` object in `config.py` (Pydantic `BaseSettings`).
- Never hardcode secrets, API keys, or connection strings.

---

## Testing

- Test runner: `uv run pytest`
- Always follow Red–Green–Refactor TDD
- Use `pytest-asyncio` with `asyncio_mode = "auto"` in `pyproject.toml`.
- **Tests are co-located with their target** — a test for `nodes/retriever.py` lives at `nodes/retriever_test.py`, not in a separate `tests/` directory.
- Test files are named `<module>_test.py` and live in the same directory as the module they test.
- Mock all external services (Supabase, OpenAI) in unit tests — never hit live APIs in tests.
- Integration tests that require a live DB are marked `@pytest.mark.integration` and skipped by default.
- Shared fixtures live in `conftest.py` at the project root (or a local `conftest.py` for directory-scoped fixtures).

---

## Linting & Formatting

Ruff is the single tool for both linting and formatting.

---

## What NOT to Do

- Do not commit `.env` (it is in `.gitignore`)
- Do not instantiate the Supabase client outside `src/db/client.py`
- Do not use `pip install` — always use `uv add`
- Do not use bare `except:` — always catch specific exceptions
