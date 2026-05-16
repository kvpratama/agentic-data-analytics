---
name: analyst
description: Analyse a dataset and produce a Markdown report with plots. Use whenever asked to analyse a dataset.
---

# Data Analysis Skill

## Overview

Read `dataset.csv` and `changes.json`, produce a Markdown report (`report.md`) with
supporting PNG plots in `plots/`. The report follows a fixed skeleton with conditional
sections — include a conditional section only if the data genuinely warrants it.

## When to Use

When asked to analyse or summarise a dataset after cleaning has already run.

## Execution Context

Use `execute_python` for all code. Kernel is persistent across calls. Split work across
multiple calls to stay under the ~10 KB output limit. Create `plots/` before saving any
figures.

## Inputs

- `dataset.csv` — the cleaned dataset
- `changes.json` — the Cleaner's decision log; read this to understand what was fixed,
  skipped, or winsorised before drawing conclusions about the data

## Workflow

1. **Call 1 — Load**: read `dataset.csv` and `changes.json` into the kernel. Print shape
   and a quick dtype summary to confirm what you're working with.
2. **Call 2 — Compute**: run the statistics and correlation checks needed to decide which
   conditional sections are warranted.
3. **Call 3+ — Plot**: generate and save each plot as a PNG. One `execute_python` call
   per plot to keep output manageable.
4. **Final call — Write**: assemble and write `report.md`.

---

## Report Structure

### Required Sections (always include)

#### 1. Summary
One short paragraph covering:
- What the dataset contains (infer from column names and values)
- Shape after cleaning (rows × columns)
- What the Cleaner changed (summarise `changes.json` in one sentence)

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

**`report.md`** — Markdown file with:
- Required sections: Summary, Key Findings, Data Quality Notes
- Conditional sections as warranted
- PNG references using relative paths: `![title](plots/filename.png)`

**`plots/`** — directory containing all referenced PNGs.

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

**One plot per `execute_python` call** — matplotlib state can bleed between figures if
multiple plots are built in one call without explicit `plt.close()`. Keeping them separate
is safer and keeps output readable.