# Data

Raw student-level data is **not included** in this repository because it contains private educational data.

The public portfolio, including the Streamlit dashboard, can be explored without access to the raw datasets. The dashboard reads only aggregated, non-student-level outputs stored in `results/`.

## Raw files required for full reproduction

To rerun the complete analytical pipeline and retrain the model locally, place the following files in this directory:

```text
data/
├── stats__module_1.csv
├── groups.csv
├── wk_media_view_sessions.csv
├──user_trainings.csv
└──trainings.csv
```

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
