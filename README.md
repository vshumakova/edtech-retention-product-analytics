# Edtech Retention Product Analytics

An end-to-end retention analytics project for an online education platform: identify the critical churn point, detect early disengagement by Day 14, prioritize students for intervention, and define how to measure incremental business impact.

> **North Star:** Course Completion Rate  
> **Decision metric:** Module 1 → Module 2 Transition Rate  
> **ML output:** D14 Churn Risk Score

---

## Business problem

Cifrium runs a four-module educational course. The largest retention loss occurs early: **36% of students are lost between Modules 1 and 2**.

The product challenge is not simply to describe churn after it happens. The goal is to identify elevated risk **while there is still time to act**.

The project therefore scores students on **Day 14 after enrollment** and turns the prediction into an operational retention workflow.

```text
Early behavior → D14 Risk Score → Prioritized intervention
                                      ↓
                              M1 → M2 transition
                                      ↓
                             Course Completion
```

## What this project covers

- **Product analytics:** funnel, cohort retention and early behavioral signals.
- **Point-in-time analytics:** features use only information available by D14.
- **Machine learning:** Logistic Regression, HistGradientBoosting and CatBoost with chronological validation.
- **Operational metrics:** Precision@Top-K, Recall@Top-K and Lift@Top-K under limited curator capacity.
- **Product activation:** risk segments and intervention playbooks.
- **Experimentation:** A/B-test design separating predictive quality from causal uplift.
- **Data product:** interactive Streamlit dashboard for retention monitoring and intervention prioritization.

## Project deliverables

| Deliverable | Description |
|---|---|
| [Business findings](docs/BUSINESS_FINDINGS.md) | Executive product conclusions, decision framework and recommendations |
| [Model report](docs/MODEL_REPORT.md) | Prediction setup, validation, metrics, model selection and limitations |
| [Experiment design](docs/EXPERIMENT_DESIGN.md) | Framework for measuring causal retention uplift |
| [Analytical notebook](notebooks/01_cifrium_early_churn_product_case.ipynb) | Full product analysis and ML workflow |
| [Streamlit dashboard](app.py) | Stakeholder-facing retention and risk interface |

## Analytical design

The prediction snapshot represents each student exactly as observable at the end of Day 14.

```text
Enrollment                    D14 score                 Future outcome
    │                            │                           │
    ├──── feature window ────────┤                           │
    │                            └──── prediction horizon ───┤
```

Behavioral features include activity frequency, active days, recency, inactivity gaps, lesson breadth, task behavior and media engagement when event timestamps are available.

Final module outcomes and whole-period aggregates are not used as predictors.

## ML evaluation

The project uses a chronological holdout: earlier enrollment cohorts are used for training and later cohorts for testing.

Model quality is evaluated with both statistical and operational metrics:

`ROC-AUC` · `PR-AUC` · `Brier Score` · `Precision@Top-K` · `Recall@Top-K` · `Lift@Top-K`

The main operational question is:

> **If curators can contact only the highest-risk 20% of students, how much future churn can the model capture?**

Final metric values are generated from the D14 pipeline and belong in the [Model Report](docs/MODEL_REPORT.md); they should not be replaced with illustrative dashboard values.

## Interactive dashboard

Run locally:

```bash
streamlit run app.py
```

The dashboard includes:

1. **Executive Overview** — funnel, KPIs and risk mix.
2. **Retention Analytics** — cohort trends and behavioral risk curves.
3. **Risk Operations** — prioritized student intervention queue.
4. **Model Quality** — ranking quality, lift and important D14 signals.
5. **Impact Simulator** — expected retention impact under different targeting and uplift assumptions.

Private student-level data is excluded from the public repository. Without prepared analytical outputs, the app opens in a clearly marked portfolio demo mode.

## Repository structure

```text
cifrium-retention-intelligence/
├── README.md
├── app.py
├── requirements.txt
├── docs/
│   ├── BUSINESS_FINDINGS.md
│   ├── MODEL_REPORT.md
│   └── EXPERIMENT_DESIGN.md
├── notebooks/
│   └── 01_cifrium_early_churn_product_case.ipynb
├── src/
│   ├── features.py
│   ├── evaluation.py
│   └── dashboard.py
├── data/
│   └── README.md
└── .streamlit/
    └── config.toml
```

## Run locally

```bash
git clone <repository-url>
cd cifrium-retention-intelligence

pip install -r requirements.txt

jupyter notebook notebooks/01_cifrium_early_churn_product_case.ipynb
streamlit run app.py
```

## Tech stack

**Python · Pandas · NumPy · scikit-learn · CatBoost · Matplotlib · Plotly · Streamlit · Jupyter**

---

### Product principle

The final deliverable is not a classifier. It is a decision system:

**Measure → Predict → Prioritize → Intervene → Experiment → Learn**
