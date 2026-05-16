---
name: cleaner
description: Clean a CSV dataset by reading profile.json. Use whenever asked to clean a dataset after profiling.
---

# Data Cleaning Skill

## Overview

Read `profile.json` produced by the Profiler, work through the `diagnosis` list, apply
fixes to the DataFrame using your own judgement, and overwrite `dataset.csv` in place.
Log every decision — including what you chose *not* to fix and why — to `changes.json`.

## When to Use

When asked to clean a dataset after profiling has already run.

## Execution Context

Use `execute_python` for all code. The kernel is persistent — `df` and any variables set
in earlier calls survive across calls. Split work across multiple calls to stay under the
~10 KB output limit per call.

## Inputs

- `profile.json` — written by the Profiler; read this first
- `dataset.csv` — the raw dataset to clean

## Workflow

1. **Call 1 — Load**: read `profile.json` and `dataset.csv` into the kernel.
2. **Call 2 — Fix**: work through each diagnosis item, apply or skip with logged reasoning.
3. **Call 3 — Write**: overwrite `dataset.csv`, write `changes.json`, print summary.

---

## How to Handle Each Diagnosis Type

For each item in `diagnosis`, apply the fix or consciously skip it. Either way, record
your decision in `changes` (see Output Contract below).

### `"drop column 'X' (Y% missing)"`
Drop the column. High missingness leaves too little signal to impute reliably.
```python
df.drop(columns=['X'], inplace=True)
```

### `"fill 'X' with median"` / `"fill 'X' with mode"`
Impute missing values. Use the statistic computed **before** any other changes to `df`
so earlier fixes don't shift the imputation value.
```python
df['X'] = df['X'].fillna(df['X'].median())   # numeric
df['X'] = df['X'].fillna(df['X'].mode()[0])  # categorical
```

### `"cast 'X' to numeric"`
Coerce the column; non-castable values become `NaN`.
```python
df['X'] = pd.to_numeric(df['X'], errors='coerce')
```
After casting, check if the newly created `NaN`s are significant (>5%). If so, add a
follow-up imputation step and log it.

### `"outliers detected in 'X' (N rows beyond 1.5×IQR) — review before deciding to clip, winsorise, or keep"`
**Use your judgement.** Consider:
- Is this column likely to contain legitimate extremes (revenue, fraud amounts, sensor
  readings)? If yes, **keep** and log your reasoning.
- Does the column name or value range suggest data entry error (e.g. age=999, salary=-1)?
  **Clip** or **winsorise** and log.
- Is the outlier count small (<1% of rows) in a column with no domain reason for extremes?
  **Winsorise** as a safe default.

Winsorise (cap at bounds without removing rows):
```python
q1, q3 = df['X'].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df['X'] = df['X'].clip(lower=lower, upper=upper)
```

### `"drop constant column 'X'"`
Drop it — a column with one unique value carries no information.
```python
df.drop(columns=['X'], inplace=True)
```

### `"'X' looks like an ID column — confirm before using as a feature"`
**Do not drop.** ID columns are useful for traceability. Log that you reviewed it and
left it intact. If the column name strongly suggests it is a surrogate key (e.g. `uuid`,
`row_id`) and has no analytical value, you may drop it — but always log the reasoning.

### `"drop N duplicate rows"`
Drop duplicates, keeping the first occurrence.
```python
df.drop_duplicates(inplace=True)
```

### `"strip whitespace in 'X' (N affected rows)"`
Strip leading/trailing whitespace in-place.
```python
df['X'] = df['X'].str.strip()
```

---

## Judgement Guidelines

The diagnosis list is a starting point, not a rigid checklist. You are expected to:

- **Re-order fixes** when needed — e.g. cast to numeric before imputing, drop columns
  before computing duplicate counts.
- **Notice interactions** — dropping a high-null column may eliminate the only rows where
  another column has nulls, making a planned imputation unnecessary.
- **Add unlisted fixes** if you spot an obvious issue the Profiler missed (e.g. a date
  column stored as object that wasn't flagged as castable). Log it as `"unlisted"`.
- **Skip items** that don't make sense given the data — always log the reason.

---

## Output Contract

**`dataset.csv`** — overwritten in place with the cleaned DataFrame.

**`changes.json`** — one entry per diagnosis item:

```json
[
  {
    "diagnosis": "fill 'age' with median",
    "action": "applied",
    "detail": "filled 12 nulls with median value 34.0"
  },
  {
    "diagnosis": "outliers detected in 'revenue' (8 rows beyond 1.5×IQR) — review...",
    "action": "skipped",
    "detail": "revenue is a financial metric where large values are legitimate; kept as-is"
  },
  {
    "diagnosis": "outliers detected in 'age' (2 rows beyond 1.5×IQR) — review...",
    "action": "winsorised",
    "detail": "age=312 and age=-4 are implausible entry errors; capped to [18, 72]"
  },
  {
    "diagnosis": "unlisted",
    "action": "applied",
    "detail": "cast 'signup_date' from object to datetime — obvious date column missed by profiler"
  }
]
```

`action` must be one of: `"applied"`, `"skipped"`, `"winsorised"`, `"unlisted"`.

Print to stdout:
```
Cleaned dataset.csv: N fixes applied, M skipped. Rows: X → Y. Cols: A → B.
```

---

## Gotcha

**Compute imputation statistics before modifying `df`**, otherwise earlier drops or casts
shift the values:

```python
# Correct — snapshot fill values first
fill_values = {
    col: df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]
    for col in cols_to_fill
}
# Then apply in a single call
df = df.fillna(fill_values)
```
