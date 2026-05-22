---
name: profiler
description: Profile a CSV dataset using pandas to surface dtypes, missingness, cardinality, distributions, outliers, and duplicates. Use whenever asked to profile a dataset before cleaning or analysis.
---

# Data Profiling Skill

## Overview

Inspect `dataset.csv` and produce `profile.json` containing both raw statistics and a
`diagnosis` list of concrete, actionable issues a downstream cleaner should address.

## When to Use

When asked to profile a dataset before cleaning or analysis.

## Execution Context

Use the `execute` shell tool to run code. Each `execute` call is a fresh Python process —
**state persists via files, not in-memory variables.** Re-read `/work/dataset.csv` at the
top of each script.

- **All dataset files live under `/work/`.** Use absolute paths: `/work/dataset.csv`,
  `/work/profile.json`.
- **Output is truncated at ~10 KB.** Split the profiling work across multiple `execute`
  calls (one per logical section) rather than submitting the entire snippet in a single
  call. This keeps each return value readable and avoids silent truncation mid-diagnosis.
- **Timeout is 60 s per `execute` call.** Wide datasets with many columns may need the
  loops broken up if they run long.

### How to run Python

For multi-line Python, write a script and run it:

```
write_file('/work/_cell.py', '<your code>')
execute('python /work/_cell.py')
```

For one-liners, `execute("python -c '...'")` is fine. Avoid heredocs — they are brittle
through the LLM's quoting.


## Workflow

1. **Call 1 — Load + raw stats**: in a script, `df = pd.read_csv('/work/dataset.csv')`, build the `profile` dict (shape, dtypes, missingness, cardinality, describe, duplicates), and print key counts.
2. **Call 2 — Diagnosis**: re-read `/work/dataset.csv`, iterate over columns to build the `diagnosis` list (missingness rules, castability, outliers, cardinality flags, duplicates, whitespace).
3. **Call 3 — Write output**: merge `diagnosis` into `profile`, `json.dump` to `/work/profile.json`, print summary.

Splitting across three calls keeps each return value well under the 10 KB output limit.

## What to Inspect

- **Shape**: `df.shape`
- **Dtypes**: `df.dtypes.astype(str).to_dict()`
- **Missingness**: `df.isna().mean()` — flag >50% as drop candidates, 5–50% as fill candidates.
- **Cardinality**: `df.nunique()` — flag `nunique == 1` (constant), `nunique == len(df)` (likely ID column).
- **Numeric distributions**: `df.describe()` (numeric cols only).
- **Outliers (IQR)**: for each numeric column, count rows outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`.
- **Duplicates**: `df.duplicated().sum()`.
- **Whitespace**: for each string column, `df[col].dropna().str.strip().ne(df[col].dropna()).sum()` (drop nulls on both sides so the count reflects only non-null mismatches).
- **Castability**: for each object column, check if `pd.to_numeric(df[col], errors='coerce')` succeeds for ≥90% of non-null values.

## Diagnosis Rules

Append one string per issue to `diagnosis`:

| Condition | Diagnosis string |
|---|---|
| Missing > 50% | `"drop column 'X' (Y% missing)"` |
| Missing 5–50%, numeric | `"fill 'X' with median"` |
| Missing 5–50%, categorical | `"fill 'X' with mode"` |
| Object col ≥90% castable to numeric | `"cast 'X' to numeric"` |
| IQR outliers > 0 | `"outliers detected in 'X' (N rows beyond 1.5×IQR) — review before deciding to clip, winsorise, or keep"` |
| `nunique == 1` | `"drop constant column 'X'"` |
| `nunique == len(df)` | `"'X' looks like an ID column — confirm before using as a feature"` |
| Duplicate rows > 0 | `"drop N duplicate rows"` |
| Whitespace detected | `"strip whitespace in 'X' (N affected rows)"` |

> **Note on thresholds**: the 5% missingness threshold is a heuristic. On very small
> datasets (<500 rows), even 5% missing may warrant manual review rather than imputation.

## Output Contract

File: `/work/profile.json`.

```json
{
  "shape": [1000, 12],
  "dtypes": {"age": "float64", "category": "object"},
  "missing_pct": {"age": 1.2, "notes": 63.0},
  "nunique": {"age": 58, "category": 5},
  "describe": { "...numeric describe() output..." },
  "duplicate_rows": 5,
  "diagnosis": [
    "drop column 'notes' (63% missing)",
    "fill 'age' with median",
    "cast 'revenue' to numeric",
    "outliers detected in 'age' (3 rows beyond 1.5×IQR) — review before deciding to clip, winsorise, or keep",
    "drop constant column 'region'",
    "'id' looks like an ID column — confirm before using as a feature",
    "drop 5 duplicate rows",
    "strip whitespace in 'category' (12 affected rows)"
  ]
}
```

Print to stdout: `"Profiled N rows × M cols; K issues found."`

## Gotchas

Two patterns that are commonly written wrong — use these exactly.

**Whitespace check** — drop nulls on both sides first so null positions don't leak into
the `.ne()` comparison and inflate (or otherwise distort) the affected-row count. The
expression below must match the short rule in *What to Inspect* exactly so counts agree:

```python
n = int(df[col].dropna().str.strip().ne(df[col].dropna()).sum())
```

**Writing profile.json** — `default=str` is required to handle numpy scalars that
`json.dump` can't serialise natively:

```python
with open('/work/profile.json', 'w') as f:
    json.dump(profile, f, indent=2, default=str)
```