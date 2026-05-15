---
name: profiling
description: Profile a CSV dataset using pandas to surface dtypes, missingness, cardinality, distributions, outliers, and duplicates, then write profile.json with raw stats and a diagnosis list.
---

# Data Profiling Skill

## Overview

You inspect `dataset.csv` and produce `profile.json` containing both raw
statistics and a `diagnosis` list of concrete issues a downstream cleaner
should address.

## When to Use

When asked to profile a dataset before cleaning or analysis.

## Workflow

1. Load the CSV once: `df = pd.read_csv('dataset.csv')`. Reuse `df` across cells.
2. Inspect: shape, dtypes, missingness, cardinality, distributions, outliers, duplicates.
3. Build a `profile` dict and a `diagnosis` list of human-readable issue strings.
4. Write the result with `json.dump(..., default=str)` to handle numpy types.
5. Print a short summary to stdout so the orchestrator sees progress.

## What to Inspect

- **Shape**: `df.shape`
- **Dtypes**: `df.dtypes.astype(str).to_dict()`
- **Missingness**: `df.isna().mean().sort_values(ascending=False)`
- **Cardinality**: `df.nunique()` — flag columns with `nunique == 1` (constant) and `nunique == len(df)` (likely IDs).
- **Numeric distributions**: `df.describe()`
- **Outliers (IQR)**: for each numeric column, count rows outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
- **Duplicates**: `df.duplicated().sum()`
- **String columns**: check for leading/trailing whitespace via `df[col].str.strip().ne(df[col]).sum()`.

## Diagnosis Rules (write these as strings into `diagnosis`)

- Missing > 50% in a column → `"drop column 'X' (Y% missing)"`
- Missing 5–50% in numeric → `"fill 'X' with median"`; in categorical → `"fill 'X' with mode"`.
- Object column where `pd.to_numeric(..., errors='coerce')` succeeds for ≥90% → `"cast 'X' to numeric"`.
- IQR-outlier count > 0 → `"clip outliers in 'X' (N rows beyond 1.5*IQR)"`.
- Constant column → `"drop constant column 'X'"`.
- `df.duplicated().sum() > 0` → `"drop N duplicate rows"`.
- Whitespace in strings → `"strip whitespace in 'X'"`.

## Output Contract

- File: `profile.json` in the work directory.
- Shape:

```json
{
  "shape": [rows, cols],
  "dtypes": {"col": "dtype", ...},
  "missing_pct": {"col": float, ...},
  "nunique": {"col": int, ...},
  "describe": { ... pandas describe() as dict ... },
  "duplicate_rows": int,
  "diagnosis": ["...", "..."]
}
```

- Print: `"Profiled N rows × M cols; K issues found."` to stdout.

## Reference Snippets

```python
import json
import pandas as pd

df = pd.read_csv('dataset.csv')

profile = {
    "shape": list(df.shape),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
    "nunique": df.nunique().to_dict(),
    "describe": df.describe(include='all').to_dict(),
    "duplicate_rows": int(df.duplicated().sum()),
}

diagnosis = []
for col, pct in profile["missing_pct"].items():
    if pct > 50:
        diagnosis.append(f"drop column '{col}' ({pct:.0f}% missing)")
    elif pct > 5:
        kind = "median" if pd.api.types.is_numeric_dtype(df[col]) else "mode"
        diagnosis.append(f"fill '{col}' with {kind}")

for col in df.select_dtypes(include='number').columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    if n:
        diagnosis.append(f"clip outliers in '{col}' ({n} rows beyond 1.5*IQR)")

if profile["duplicate_rows"]:
    diagnosis.append(f"drop {profile['duplicate_rows']} duplicate rows")

profile["diagnosis"] = diagnosis

with open('profile.json', 'w') as f:
    json.dump(profile, f, indent=2, default=str)

print(f"Profiled {df.shape[0]} rows × {df.shape[1]} cols; {len(diagnosis)} issues found.")
```
