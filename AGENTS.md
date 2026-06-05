## Project Overview

Multi-subagent EDA orchestrator using Deep Agents and ephemeral Modal sandboxes. The system coordinates three specialized subagents to perform end-to-end data analysis:
1. **Profiler**: Understands the dataset structure, types, and summary statistics.
2. **Cleaner**: Fixes data quality issues based on the profile (handling missing values, types, etc.).
3. **Analyst**: Performs deep-dive analysis, generates visualizations, and writes the final report.

All subagents share a single Modal sandbox (ephemeral microVM) per turn, coordinating through explicit files under `/workspace/` (e.g. `profile.json`, `dataset.clean.csv`).

---

## Tech Stack

- **Runtime**: Python 3.12+
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) & [LangChain Deep Agents](https://github.com/langchain-ai/deepagents)
- **Execution**: `langchain-modal` (`ModalSandbox`) for sandboxed shell execution in ephemeral Modal microVMs
- **Data Science**: `pandas`, `scipy`, `matplotlib`, `seaborn`
- **Models**: Anthropic (default), OpenAI, or Google GenAI via LangChain `init_chat_model`
- **Environment**: `uv` for dependency management, `pydantic-settings` for configuration

---

## Project Structure

- `ada/`: Main Python package directory
  - `agent.py`: LangGraph factory (`make_graph`) and orchestrator (`create_analytics_agent`).
  - `agent_test.py`: Unit tests for agent.
  - `subagents.py`: Subagent definitions (profiler, cleaner, analyst).
  - `cli.py`: CLI entrypoint (`ada` command — interactive REPL and one-shot mode).
  - `cli_test.py`: Unit tests for CLI.
  - `agent_middleware.py`: `SandboxLifecycleMiddleware` — mirrors `/workspace/` artifacts and terminates sandboxes.
  - `agent_middleware_test.py`: Unit tests for middleware.
  - `config.py`: Centralized configuration (`Settings`) and model initialization (`get_model`).
  - `config_test.py`: Unit tests for Settings.
  - `runtime/`: Modal sandbox runtime helpers and workspace logic.
    - `modal_runtime.py`: Sandbox image build, seeding, and artifact download.
    - `modal_runtime_test.py`: Unit tests for sandbox runtime.
    - `workspace.py`: Workspace mirroring and sandbox provisioning (`provision_workspace`).
    - `workspace_test.py`: Unit tests for workspace provisioning.
- `skills/`: Domain-specific `SKILL.md` files for profiler, cleaner, analyst, and orchestrator roles.
- `workspace/`: Gitignored, host-side per-thread artifact mirrors.
- `dataset/`: Gitignored, CSV files downloaded on demand.
- `langgraph.json`: LangGraph Studio/API graph entrypoint.
- `.env.example`: Example environment variable configuration.
- `.pre-commit-config.yaml`: Pre-commit hooks configuration.
- `.github/workflows/ci.yml`: GitHub Actions CI (lint, typecheck, test on Python 3.12/3.13).
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
- **Async-first where applicable** — use `async def` for I/O-bound functions (Modal sandbox calls, LLM calls, file operations).
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
- **Tests are co-located with their target** — a test for `config.py` lives at `config_test.py`, a test for `runtime/workspace.py` lives at `runtime/workspace_test.py`, not in a separate `tests/` directory.
- Test files are named `<module>_test.py` and live in the same directory as the module they test.
- Mock all external services (Modal, Anthropic/OpenAI/Google) in unit tests — never hit live APIs in tests.
- Integration tests that require a live sandbox are marked `@pytest.mark.integration` and skipped by default.
- Shared fixtures live in `conftest.py` at the project root (or a local `conftest.py` for directory-scoped fixtures).

---

## Linting & Formatting

Ruff is the single tool for both linting and formatting.

---

## What NOT to Do

- Do not commit `.env` (it is in `.gitignore`)
- Do not use `pip install` — always use `uv add`
- Do not use bare `except:` — always catch specific exceptions
