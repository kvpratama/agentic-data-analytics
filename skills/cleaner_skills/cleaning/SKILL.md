---
name: cleaning
description: Clean a pandas DataFrame in place using the diagnosis from profile.json, applying missing-value handling, dtype coercion, outlier clipping, duplicate removal, and string normalization, then overwrite dataset.csv.
---

# Data Cleaning Skill

## Overview

Read `profile.json` first. Apply targeted fixes to `dataset.csv` based on the
profiler's `diagnosis`, plus your own judgement. Overwrite `dataset.csv` with
the cleaned frame and print a changelog.

## When to Use

When asked to clean a dataset after profiling.

## Workflow

1. Load `profile.json` and `dataset.csv`.
2. Walk the `diagnosis` list — each entry hints at one operation.
3. Apply operations cumulatively to `df`. Maintain a `changelog` list of strings.
4. Overwrite the CSV: `df.to_csv('dataset.csv', index=False)`.
5. Print the changelog.

## Decision Rules

| Issue | Action |
|---|---|
| `> 50%` missing | `df.drop(columns=[col])` |
| 5–50% missing, numeric | `df[col].fillna(df[col].median(), inplace=True)` |
| 5–50% missing, categorical | `df[col].fillna(df[col].mode()[0], inplace=True)` |
| Object column ≥90% coercible to number | `df[col] = pd.to_numeric(df[col], errors='coerce')` then re-handle missingness |
| IQR outliers in numeric column | `df[col] = df[col].clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)` |
| Constant column | `df.drop(columns=[col])` |
| Duplicate rows | `df = df.drop_duplicates()` |
| Whitespace in strings | `df[col] = df[col].str.strip()` |

Use your judgement when rules conflict (e.g., drop high-missing column before computing its median).

## Output Contract

- File: `dataset.csv` (overwrite, no index).
- Print: a `Cleanup changelog:` header followed by one line per operation.

## Reference Snippets

```python
import json
import pandas as pd

with open('profile.json') as f:
    profile = json.load(f)

df = pd.read_csv('dataset.csv')
changelog = []

# Drop high-missing columns first
for col, pct in profile['missing_pct'].items():
    if pct > 50 and col in df.columns:
        df = df.drop(columns=[col])
        changelog.append(f"dropped '{col}' ({pct:.0f}% missing)")

# Fill remaining missingness
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            v = df[col].median()
            df[col] = df[col].fillna(v)
            changelog.append(f"filled '{col}' missing with median ({v})")
        else:
            v = df[col].mode().iloc[0]
            df[col] = df[col].fillna(v)
            changelog.append(f"filled '{col}' missing with mode ('{v}')")

# Clip numeric outliers
for col in df.select_dtypes(include='number').columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    if n:
        df[col] = df[col].clip(lower=lo, upper=hi)
        changelog.append(f"clipped {n} outliers in '{col}'")

# Drop duplicates
before = len(df)
df = df.drop_duplicates()
dropped = before - len(df)
if dropped:
    changelog.append(f"dropped {dropped} duplicate rows")

# Strip whitespace in object columns
for col in df.select_dtypes(include='object').columns:
    stripped = df[col].astype(str).str.strip()
    if (stripped != df[col]).any():
        df[col] = stripped
        changelog.append(f"stripped whitespace in '{col}'")

df.to_csv('dataset.csv', index=False)

print("Cleanup changelog:")
for line in changelog:
    print(f"  - {line}")
```
