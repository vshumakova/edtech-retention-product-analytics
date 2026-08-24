# Data

Raw student-level data is **not included** in this repository because it contains private educational data.

The public portfolio, including the Streamlit dashboard, can be explored without access to the raw datasets. The dashboard reads only aggregated, non-student-level outputs stored in `results/`.

## Raw files required for full reproduction

To rerun the complete analytical pipeline and retrain the model locally, place the following files in this directory:

```text
data/
├── stats__module_1.csv
├── groups.csv
└── wk_media_view_sessions.csv
```

### `stats__module_1.csv`

Student-level Module 1 data used to define the analytical population and the matured churn target.

The final student status is used **only as the target variable**. Final progress and whole-module outcome fields are not used as D14 predictors.

### `groups.csv`

Schedule data used to reconstruct the actual start of learning.

The project maps:

```text
stats.id параллели
        ↕
groups.group_template_id
        ↓
min(groups.starts_at) = learning_start
```

This is important because the administrative enrollment date can precede the actual course start by several weeks.

### `wk_media_view_sessions.csv`

Timestamped media-viewing events used to construct early behavioral features.

Only events inside the observation window are eligible:

```text
learning_start <= event_timestamp <= learning_start + 14 days
```

This point-in-time restriction is a core part of the leakage-prevention design.

## Public dashboard data

The Streamlit application does **not** read files from `data/`.

It uses public-safe aggregated outputs:

```text
results/
├── model_metrics.csv
├── temporal_validation.csv
├── behavioral_summary.csv
├── feature_importance.csv
├── funnel_summary.csv
├── capacity_curve.csv
└── cohort_performance.csv
```

These files contain model- and cohort-level summaries rather than individual student records.

## Privacy and Git

Raw datasets should remain local and must not be committed to the repository.

The project `.gitignore` excludes common raw-data formats from `data/`, while keeping this documentation file under version control.

Before committing, it is still good practice to verify:

```bash
git status
```

No private student-level files should appear in the staged changes.

## Reproducibility

With the private source files available locally, the full analysis can be rerun from:

```bash
jupyter notebook notebooks/01_cifrium_early_churn_product_case.ipynb
```

For portfolio review or dashboard deployment, the raw datasets are not required:

```bash
streamlit run app.py
```
