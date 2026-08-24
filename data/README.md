# Data

Source files are intentionally not committed.

Expected source tables from the Cifrium export:
- `stats__module_1.csv`
- `stats__module_2.csv`
- `stats__module_3.csv`
- `stats__module_4.csv`
- `user_activity_histories.csv`
- `user_lessons.csv`
- `user_answers.csv`
- `wk_media_view_sessions.csv`
- optionally other raw behavioral tables.

## Point-in-time requirement

A behavioral table is eligible for D14 feature engineering only if it contains an
event timestamp that can prove the event happened no later than 14 days after enrollment.

Final snapshots and outcome fields must not be used as predictors.


## Optional prepared files for the Streamlit app

To render real project results instead of portfolio demo data, place:

- `dashboard_funnel.csv`
- `dashboard_cohorts.csv`
- `dashboard_students.csv`
- `dashboard_model_metrics.csv`
- `dashboard_feature_importance.csv`

These are derived outputs, not raw source data.
