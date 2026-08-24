# Cifrium — Early Churn Detection at Day 14

**Product Analytics + Machine Learning case**

Cifrium is an online education platform with a four-module course. The key retention problem is concentrated at the beginning of the journey: **36% of students are lost between modules 1 and 2**.

The project designs an early-warning system that scores every student on **day 14 after enrollment**, while there is still time to intervene.

## Product goal

**North Star Metric:** Course Completion Rate  
Share of enrolled students who complete the course and pass the final assessment in module 4.

**Decision metric:** Module 1 → Module 2 transition rate.

**ML output:** D14 Churn Risk Score.

The score is not the product by itself. Its purpose is to prioritize interventions — curator outreach, study-plan support, content nudges — and ultimately improve retention and Course Completion Rate.

---

## Executive product logic

```text
D14 behavioral signals
        ↓
Churn Risk Score
        ↓
prioritized intervention
        ↓
higher M1 → M2 transition
        ↓
higher Course Completion Rate
```

The analytical unit is **student × scoring date**.

Only information that was actually available by the end of day 14 can be used in the model.

---

## 1. Retention diagnosis

The course consists of four modules, but churn is not evenly distributed.

The first product question is therefore not “which algorithm should predict churn?”, but:

> **At which stage of the learning journey do we lose the most students, and early enough for which behaviors can we still influence the outcome?**

The project reconstructs the module funnel and compares:
- enrollment volume by module;
- completion / churn by module;
- transition from module 1 to module 2;
- behavioral differences visible during the first 14 days.

The main intervention point is the first module.

---

## 2. Point-in-time feature design

The scoring cutoff is:

```text
cutoff = enrollment_timestamp + 14 days
```

All behavioral events must satisfy:

```text
enrollment_timestamp <= event_timestamp <= cutoff
```

### Feature families

**Learning activity**
- number of LMS actions;
- active days;
- sessions;
- recency;
- longest inactivity gap;
- lesson diversity;
- video / notes / translation behavior.

**Task behavior**
- number of attempts by D14;
- unique tasks attempted;
- solved rate based only on timestamped answers;
- persistence: attempts per task;
- skipped-task share.

**Media engagement**
- media sessions by D14;
- average watch depth;
- share of sessions watched ≥80%;
- live-session exposure if timestamped.

**Access / cadence**
- login / access events;
- activity concentration;
- weekday / weekend behavior;
- delay to first activity.

### Variables not used as D14 predictors

Final module snapshots, assessment outcomes and any aggregate whose value is not timestamped at or before D14 are excluded from the feature matrix.

Examples:
- final solved-task totals;
- “all required tasks completed” flags;
- final assessment score / result;
- final module status;
- whole-period training totals.

The project uses these fields only where appropriate as outcomes or descriptive end-state metrics.

---

## 3. Modeling strategy

A good churn model must generalize to **future cohorts**, not randomly held-out rows from the same period.

Validation therefore uses a chronological holdout:
- earlier enrollment cohorts → train;
- later enrollment cohorts → test.

Models:
1. Logistic Regression — interpretable baseline;
2. HistGradientBoosting — strong non-linear sklearn benchmark;
3. CatBoost — strong tabular model when available.

The final model is selected on a combination of:
- ROC-AUC;
- PR-AUC;
- Brier score / calibration;
- Recall@Top-K;
- Precision@Top-K;
- Lift@Top-K.

For the product team, **Recall@Top-K** is especially important because curator capacity is finite.

---

## 4. Product decision layer

Suppose curators can personally contact only 15–20% of a cohort.

The operational question becomes:

> Among the top 20% highest-risk students, what share of future churners do we capture?

Example policy:

| Risk segment | Action |
|---|---|
| High | Personal curator outreach within 24 hours |
| Medium | Automated nudge + study-plan recommendation |
| Low | Standard course journey |

The threshold is therefore chosen from **capacity and expected uplift**, not from a generic probability threshold of 0.5.

---

## 5. Experiment design

Offline ML quality does not prove retention impact.

The production hypothesis is:

> A targeted intervention for high-risk D14 students increases module-2 transition and, downstream, Course Completion Rate.

Recommended A/B design:
- eligibility: students in the high-risk segment;
- control: business-as-usual;
- treatment: intervention;
- primary experiment metric: M1 → M2 transition;
- secondary: Course Completion Rate;
- guardrails: complaints / opt-outs, curator workload, intervention cost.

### Impact decomposition

```text
incremental retained students
= contacted students
× true-risk concentration
× intervention uplift
```

This separates three different problems:
1. ranking quality of the model;
2. targeting capacity;
3. causal effectiveness of the intervention.

---

## 6. Repository

```text
cifrium-early-churn/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── 01_cifrium_early_churn_product_case.ipynb
└── src/
    ├── features.py
    └── evaluation.py
```

---

## 7. Run locally

Place the source CSV files in `data/` and run:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_cifrium_early_churn_product_case.ipynb
```

The notebook checks timestamp availability before using a behavioral table. If an event cannot be located in time relative to D14, it is not silently included.

---

## 8. What makes this a product analytics project

The core deliverable is not a classifier leaderboard.

The project connects:
- **funnel diagnosis** → where the retention problem is;
- **behavioral analytics** → what early signals distinguish at-risk students;
- **predictive modeling** → who should be prioritized;
- **capacity-aware targeting** → whom the team can actually contact;
- **experimentation** → whether intervention changes behavior;
- **North Star impact** → whether retention improvement reaches course completion.

That is the intended production loop.


---

## Interactive Streamlit dashboard

The repository includes an interactive product dashboard:

```bash
streamlit run app.py
```

The dashboard has five views:

- **Executive overview** — key retention problem, funnel, risk mix and operating logic;
- **Retention analytics** — cohort retention and behavioral risk curves;
- **Risk operations** — ranked D14 student queue with recommended interventions;
- **Model quality** — ROC-AUC, PR-AUC, Recall@Top-K, Precision@Top-K, Lift and feature importance;
- **Impact simulator** — translates targeting quality and causal intervention uplift into expected retained students and Course Completion impact.

### Portfolio demo mode

Private student-level data is not committed to GitHub. If prepared dashboard exports are absent, the app automatically opens in a clearly marked **demo mode** with illustrative data so that the interface remains deployable for portfolio review.

To connect real outputs, export these files into `data/`:

```text
dashboard_funnel.csv
dashboard_cohorts.csv
dashboard_students.csv
dashboard_model_metrics.csv
dashboard_feature_importance.csv
```

The helper `src/dashboard.py` contains the export function.

### Recommended GitHub / Streamlit Cloud setup

1. Push the repository to GitHub.
2. Keep raw/private educational data out of the repository.
3. Deploy `app.py` on Streamlit Community Cloud.
4. Add the live dashboard link near the top of this README.
5. Keep the notebook as the analytical narrative and `src/` as reusable production-style code.

---

## Why the repository contains both `.ipynb` and `.py`

This is intentional.

- `notebooks/01_cifrium_early_churn_product_case.ipynb` is the **analytical story**: product framing, EDA, cohort analysis, feature validation, model comparison and conclusions.
- `src/*.py` contains **reusable logic**: feature engineering, evaluation and dashboard exports.
- `app.py` is the **presentation / decision layer** for stakeholders.
- `README.md` is the **portfolio landing page**.

For a data / product analytics portfolio, this structure is stronger than either a notebook-only repository or a collection of Python scripts with no analytical narrative.
