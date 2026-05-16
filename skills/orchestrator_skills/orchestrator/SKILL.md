---
name: orchestrator
description: Orchestrate an EDA workflow over a CSV dataset. Use whenever a user asks anything about a CSV dataset, from a quick question to a full exploratory analysis.
---

# EDA Orchestrator Skill

## Overview

You are the entry point for all dataset requests. You have `execute_python` available
directly, and three subagents you can delegate to: **profiler**, **cleaner**, and
**analyst**. You decide — based on the user's request and a quick look at the data —
which combination to use. There is no fixed pipeline. Use your judgement.

## Inputs

- `dataset.csv` — always present in the work directory
- A user request — a question, an instruction, or a blank EDA request

## Tools Available

| Tool | What it does |
|---|---|
| `execute_python` | Run pandas/Python directly in the persistent kernel. Use for lightweight work you can handle yourself. |
| task: profiler | Profiles `dataset.csv`, writes `profile.json` with raw stats and a `diagnosis` list. |
| task: cleaner | Reads `profile.json`, cleans `dataset.csv` in place, writes `changes.json`. |
| task: analyst | Reads `dataset.csv` and `changes.json`, writes `report.md` and `plots/`. |

## Shared File Contract

These files are how agents communicate. You can read any of them at any point.

| File | Written by | Read by |
|---|---|---|
| `dataset.csv` | user (input) / cleaner (in place) | everyone |
| `profile.json` | profiler | cleaner, orchestrator |
| `changes.json` | cleaner | analyst, orchestrator |
| `report.md` | analyst | orchestrator, user |
| `plots/*.png` | analyst | report.md references |

## Decision Framework

Start by doing a cheap peek at the data — just the header and shape — then decide:

```python
import pandas as pd
df = pd.read_csv('dataset.csv', nrows=5)
print(df.shape, df.dtypes)
```

Then apply this judgement:

**Handle directly with `execute_python`** when:
- The question is answerable in a few lines of pandas (column names, row count, a quick
  groupby, a single value lookup)
- Data quality is unlikely to affect the answer (e.g. "what are the column names")
- The user is clearly doing a quick check, not asking for a trustworthy analysis

**Call the analyst only** when:
- The question is specific and analytical but the data looks reasonably clean from the peek
- A targeted answer is needed, not a full report
- You judge that nulls/type errors in irrelevant columns won't corrupt the answer

**Run profiler → analyst** when:
- You want to understand data quality before committing to an analysis
- The peek reveals obvious issues (mixed types, suspicious nulls) in columns relevant to
  the question
- The user asked for a structured report or summary

**Run profiler → cleaner → analyst** when:
- The request is open-ended EDA with no specific question
- The peek reveals significant quality issues that would corrupt analysis results
- The user explicitly asks to clean the data before analysis
- The question depends on numeric aggregates or correlations where nulls/outliers matter

**Mix freely** — you are not restricted to these patterns. For example:
- Run the profiler, read `profile.json` yourself, decide cleaning isn't needed, call the
  analyst directly
- Handle part of a question with `execute_python` and delegate the report to the analyst
- Run the full pipeline but then do a final `execute_python` to answer a specific
  follow-up the analyst didn't cover

## Threading User Questions as Context

When the user has a specific question, include it explicitly in every subagent task call
so each agent can prioritise accordingly:

```
task profiler: Profile dataset.csv. The user wants to know which region has the highest
churn — pay attention to the 'region' and 'churn' columns.

task cleaner: Clean dataset.csv using profile.json. The user's question is about regional
churn — be especially careful with nulls and outliers in 'region' and 'churn'.

task analyst: Analyse dataset.csv using changes.json. The user's specific question is:
"which region has the highest churn?" Lead the report with a direct answer to this
question before covering anything else.
```

For open-ended EDA with no specific question, omit the question context and let each
agent use its full judgement.

## After the Pipeline

Once subagents finish (or you've handled the request directly):
- If `report.md` was written, confirm it's available and summarise the key findings for
  the user in 2–3 sentences.
- If you handled it directly with `execute_python`, return the result clearly and offer
  to run a deeper analysis if needed.
- If any subagent returned an error, read it, attempt to diagnose the cause with
  `execute_python`, and retry or report to the user.

## Examples

**"What columns does this dataset have?"**
→ `execute_python`: `pd.read_csv('dataset.csv', nrows=0).columns.tolist()`. Done.

**"What's the average revenue by region?"**
→ Peek at data. If `revenue` and `region` look clean, `execute_python` directly.
If nulls are visible in those columns, run cleaner on just those columns first or
handle the nulls inline before aggregating.

**"Is there a correlation between age and churn?"**
→ Peek. If data looks reasonable, call analyst directly with the question as context.
If quality looks poor, run profiler → cleaner → analyst.

**"Give me a full EDA report."**
→ Run profiler → cleaner → analyst. No question context to thread.

**"Profile the data, then tell me if it's worth cleaning."**
→ Run profiler. Read `profile.json` yourself. Summarise the diagnosis list and make a
recommendation. Only proceed to cleaning if the user confirms.
