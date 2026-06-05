# LangGraph EDA Orchestrator (`ada_lg`)

This sub-package contains a parallel implementation of the Agentic Data Analytics (`ada`) orchestrator. While the main `ada` package uses LangChain's `deepagents` framework, `ada_lg` uses **pure LangGraph** (standard state graphs compiled via `langgraph`) to coordinate the multi-subagent workflow.

## Overview

The pure-LangGraph implementation mirrors the functionality of the reference `ada` agent, including:
1. **Profiler**: Inspects raw data and writes a profile diagnosis.
2. **Cleaner**: Resolves data quality issues listed in the profile.
3. **Analyst**: Generates visualizations and produces the final Markdown report.

All subagents share an ephemeral `ModalSandbox` (secure microVM container) and coordinate via file state handoff in `/workspace/`.

## Architecture & Code Structure

The directory is structured as follows:
- [`__init__.py`](./__init__.py): Package initialization.
- [`agent.py`](./agent.py): LangGraph agent factory and execution provisioning flow.
- [`subagents.py`](./subagents.py): Defines system prompts and skill scopes for the three subagents.
- [`cli.py`](./cli.py): Sibling CLI wrapper that monkeypatches `ada.cli` to run the LangGraph orchestrator instead.
- [`lg/`](./lg/): Custom LangGraph tool definitions and factory methods.
  - [`agent_factory.py`](./lg/agent_factory.py): Assembles LangGraph subagents and the main orchestrator using `create_agent`.
  - [`fs_tools.py`](./lg/fs_tools.py): Wraps the deep agents `BackendProtocol` to expose standard filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `execute`).
  - [`permissions.py`](./lg/permissions.py): Provides permission wrappers to prevent agents from writing to protected paths (e.g., `/skills/`).
  - [`skills.py`](./lg/skills.py): Implements progressive-disclosure skills scanning and loads them on demand via a `read_skill` tool.
  - [`task_tool.py`](./lg/task_tool.py): Implements a `task(agent, instruction)` tool allowing the orchestrator to delegate work to compiled subagent graphs.
  - [`todos_tool.py`](./lg/todos_tool.py): Defines state schema (`AnalyticsState`) containing a `todos` list updated via the `write_todos` tool.

## Key Differences from Reference `ada`

| Feature | Reference `ada` | Pure-LangGraph `ada_lg` |
|---|---|---|
| **Underlying Framework** | `deepagents.create_deep_agent` | Pure LangGraph compiled state graphs via `create_agent` |
| **Subagent Execution** | Native `deepagents` multi-subagent orchestration | Subagents are compiled subgraphs called via a custom `task` tool |
| **Tool Set** | Auto-injected filesystem and execute tools | Explicitly built and guarded tools mapping to deep agents `BackendProtocol` |
| **Planning/Todos** | In-framework planning features | Custom planning state (`AnalyticsState`) with reducer and `write_todos` tool |

## How to Run

Running `ada_lg` is identical to running `ada`. All parameters, shell variables, and environment configuration files are shared.

### Interactive REPL
To start the interactive prompt session utilizing the LangGraph orchestrator:
```bash
uv run ada-lg
```

### One-Shot Execution
To run a one-shot turn with a specific CSV file and prompt:
```bash
uv run ada-lg --csv dataset/Titanic-Dataset.csv -p "Investigate factors that affected survival"
```

### LangGraph Studio / Dev Server
The graph can be loaded in LangGraph Studio/dev server. In [`langgraph.json`](../langgraph.json), the graphs are configured. To run the dev server:
```bash
uv run langgraph dev
```

## Running Tests

All unit tests for `ada_lg` are co-located in the same directory. You can run all unit tests in the project, which will include the `ada_lg` suite:
```bash
uv run pytest
```
