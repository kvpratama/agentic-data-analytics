---
name: analysis
description: Analyze a cleaned dataset with pandas, scipy, and matplotlib to answer the user's objective, generating plots in plots/ and a final report.md with findings and caveats.
---

# Data Analysis Skill

## Overview

Read the cleaned `dataset.csv`, run analyses tied to the user's objective,
save plots to `plots/`, and write a final `report.md`.

## When to Use

When asked to analyze a dataset after cleaning.

## Workflow

1. Load `dataset.csv`. Re-confirm shape and dtypes.
2. Re-read the user's objective from the orchestrator's task description.
3. Decide which analyses serve the objective:
   - Numeric ↔ numeric: correlations, scatter plots.
   - Categorical ↔ numeric: group-by mean/std/count, box plots, t-tests / ANOVA.
   - Categorical ↔ categorical: contingency tables, chi-squared.
4. Save every plot under `plots/` with a descriptive filename.
5. Always call `plt.close()` after `plt.savefig` to release memory.
6. Write `report.md` with sections: Overview, Findings, Charts, Caveats.

## Decision Heuristics

- Don't dump full DataFrames; summarize with `.describe()` or `.head(10)`.
- Tie every chart to a finding sentence in the report.
- State sample size and any filtering you applied.
- Note assumption violations (non-normality, unequal variances) in Caveats.

## Output Contract

- Directory: `plots/` containing PNGs.
- File: `report.md` with the four sections above. Each chart is referenced inline as `![title](plots/filename.png)`.

## Reference Snippets

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs('plots', exist_ok=True)
df = pd.read_csv('dataset.csv')

# Correlations
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr)), corr.columns)
plt.colorbar()
plt.title('Numeric correlations')
plt.tight_layout()
plt.savefig('plots/correlations.png')
plt.close()

# Group-by aggregation
grouped = df.groupby('Sex')['Survived'].agg(['mean', 'std', 'count'])
print(grouped)

# Hypothesis test (two-sample t-test)
groups = [g['Age'].dropna().values for _, g in df.groupby('Survived')]
t, p = stats.ttest_ind(*groups, equal_var=False)
print(f"Welch t-test on Age by Survived: t={t:.3f}, p={p:.4f}")

# Box plot
plt.figure(figsize=(6, 4))
df.boxplot(column='Age', by='Survived')
plt.title('Age by survival')
plt.suptitle('')
plt.savefig('plots/age_by_survival.png')
plt.close()
```

## Report Template

```markdown
# Analysis Report

## Overview
- Dataset: <rows> rows × <cols> columns after cleaning.
- Objective: <restate user objective>.

## Findings
1. <finding 1, with numbers>
2. <finding 2, with numbers>
...

## Charts
![Correlations](plots/correlations.png)
![Age by survival](plots/age_by_survival.png)

## Caveats
- <assumption violations, sample-size limits, confounders>
```
