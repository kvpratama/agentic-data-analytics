---
name: analyst
description: Analyse a dataset and produce a Markdown report with plots. Use whenever asked to analyse a dataset.
---

# Data Analysis Skill

## Overview

Read `/work/dataset.clean.csv` (or `/work/dataset.csv` if the cleaner did not run) and
`/work/changes.json`, produce a Markdown report (`/work/report.md`) with supporting PNG
plots in `/work/plots/`. The report follows a fixed skeleton with conditional sections —
include a conditional section only if the data genuinely warrants it.

The raw `/work/dataset.csv` is always available — consult it when you need to compare
pre/post-cleaning values (e.g. to verify outlier handling or show the impact of imputation).

## When to Use

When asked to analyse or summarise a dataset after cleaning has already run.

## Execution Context

Use the `execute` shell tool for all code. Each `execute` call is a fresh Python
process — **state persists via files, not in-memory variables.** Re-read the cleaned
dataset at the top of each script. Split work across multiple calls to stay under the
~10 KB output limit. Create `/work/plots/` (e.g. `os.makedirs('/work/plots',
exist_ok=True)`) before saving any figures.

Use this helper at the top of every script to pick the right file:

```python
import os
DATA = '/work/dataset.clean.csv' if os.path.exists('/work/dataset.clean.csv') else '/work/dataset.csv'
df = pd.read_csv(DATA)
```

### How to run Python

For multi-line Python, write a script and run it:

```
write_file('/work/_cell.py', '<your code>')
execute('python /work/_cell.py')
```

For one-liners, `execute("python -c '...'")` is fine. Avoid heredocs — they are brittle
through the LLM's quoting.


## Inputs

- `/work/dataset.clean.csv` — the cleaned dataset (preferred input).
- `/work/dataset.csv` — the raw, untouched dataset. Use as a fallback when
  `dataset.clean.csv` doesn't exist, or for pre/post comparisons.
- `/work/changes.json` — the Cleaner's decision log; read this to understand what was
  fixed, skipped, or winsorised before drawing conclusions about the data.
  **May not exist** if the orchestrator skipped the cleaning step. Treat its absence as
  "dataset is raw — no cleaning performed" and proceed; do not error out.

## Workflow

1. **Call 1 — Load**: in a script, read the cleaned dataset (see helper in Execution
   Context) and `/work/changes.json` if it exists (use `os.path.exists` — set
   `changes = []` if missing). Print shape and a quick dtype summary to confirm what
   you're working with.
2. **Call 2 — Compute**: re-read the cleaned dataset, run the statistics and correlation
   checks needed to decide which conditional sections are warranted.
3. **Call 3+ — Plot**: generate and save each plot as a PNG under `/work/plots/`. One
   `execute` call per plot to keep output manageable.
4. **Final call — Write**: assemble and write `/work/report.md`.

---

## Report Structure

### Required Sections (always include)

#### 1. Summary
One short paragraph covering:
- What the dataset contains (infer from column names and values)
- Shape after cleaning (rows × columns)
- What the Cleaner changed (summarise `changes.json` in one sentence, or state
  "no cleaning was performed" if `changes.json` is absent)

#### 2. Key Findings
3–5 bullet points in plain language. These should be the most actionable or surprising
observations — not restatements of statistics. Examples of good findings:
- "Customers aged 45–60 account for 60% of revenue despite being 30% of the user base."
- "Churn rate is 3× higher in the West region than any other."
- "Price and quantity_sold have a strong negative correlation (r = −0.81)."

Bad findings (too generic — avoid):
- "The dataset has 1,000 rows and 12 columns."
- "Some columns had missing values."

#### 3. Data Quality Notes
A brief table summarising what the Cleaner did, drawn from `changes.json`:

| Column | Action | Detail |
|--------|--------|--------|
| age | winsorised | capped implausible values |
| revenue | skipped | legitimate extremes kept |

Keep this section short — it's context, not analysis.

If no cleaning was performed, replace the table with a single sentence:
"Dataset analysed as-is; no cleaning step was run."

---

### Conditional Sections (include only if warranted)

#### Distributions
**Include if**: any numeric column has skewness > 1.5 or < −1.5, or a visible bimodal
pattern, or a categorical column has a dominant value (>70% one category).

For each interesting column, save a plot and reference it:
```
![age distribution](plots/dist_age.png)
```

Describe what the shape means, not just what it is. "Age is right-skewed, driven by a
small group of users over 70" is useful. "Age has skewness 1.8" is not.

#### Correlations
**Include if**: there are ≥2 numeric columns and at least one pair has |r| > 0.5.

Save a heatmap as `plots/correlation_heatmap.png`. Call out the strongest relationships
by name and suggest what they might imply.

#### Categorical Breakdown
**Include if**: there are categorical columns with 2–20 unique values that show meaningful
variation (i.e. not roughly uniform, not near-constant).

Save a bar chart per interesting column as `plots/cat_<colname>.png`.

#### Outlier Notes
**Include if**: `changes.json` contains any entry with `"action": "skipped"` for an
outlier diagnosis. Briefly explain which columns have kept outliers and why the Cleaner
left them, so the reader knows to treat those columns carefully.

---

## Plotting Guidelines

- Use `matplotlib` with `Agg` backend (no display): set `matplotlib.use('Agg')` before importing `pyplot`.
- Create `plots/` if it doesn't exist: `os.makedirs('plots', exist_ok=True)`.
- Keep plots clean: label axes, add a title, call `plt.tight_layout()` before saving.
- Save at 150 dpi: `plt.savefig('plots/filename.png', dpi=150, bbox_inches='tight')`.
- Call `plt.close()` after each save to free memory.

---

## Output Contract

**`/work/report.md`** — Markdown file with:
- Required sections: Summary, Key Findings, Data Quality Notes
- Conditional sections as warranted
- PNG references using relative paths: `![title](plots/filename.png)` (relative so the
  links still work after artifacts are mirrored back to `work/<stem>/` on the host)

**`/work/plots/`** — directory containing all referenced PNGs.

Print to stdout:
```
Report written: N sections, M plots. See report.md.
```

---

## Gotchas

**Set the matplotlib backend before importing pyplot**, otherwise it may try to open a
display and crash in a headless kernel:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # import AFTER setting backend
```

**One plot per `execute` call** — matplotlib state can bleed between figures if multiple
plots are built in one call without explicit `plt.close()`. Keeping them separate is
safer and keeps output readable.