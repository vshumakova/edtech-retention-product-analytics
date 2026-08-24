# Cifrium Retention Intelligence

**Product Analytics · Machine Learning · Experimentation · Streamlit**

An end-to-end retention analytics project for an online education platform. The goal is to identify the main retention bottleneck, detect early disengagement by Day 14 of learning, prioritize students for proactive intervention, and define how the business should measure causal impact.

> **North Star Metric:** Course Completion Rate  
> **Leading retention metric:** Module 1 → Module 2 Transition Rate  
> **ML output:** D14 Churn Risk Score

---

## Business Problem

The largest retention loss occurs at the beginning of the course: **36% of students are lost between Module 1 and Module 2**.

This makes the first module the highest-value intervention point. The product objective is to identify elevated churn risk early enough for the retention team to act.

```text
Early learning behavior
        ↓
D14 Churn Risk Score
        ↓
Prioritized intervention
        ↓
Module 1 → Module 2 transition
        ↓
Course Completion Rate
```

---

## Key Results

The final analytical sample contains **2,972 students** with a valid Module 1 learning-start date. Model performance is evaluated on a chronological future-cohort holdout of **615 students**.

### Selected operating model: Random Forest

| Metric | Result |
|---|---:|
| ROC-AUC | **0.906** |
| PR-AUC | **0.797** |
| Precision@Top-20% | **82.1%** |
| Recall@Top-20% | **53.7%** |
| Lift@Top-20% | **2.69×** |

**Product interpretation:** if the retention team can contact only the highest-risk 20% of students, the model captures **53.7% of all future churners**. Churn is **2.69× more concentrated** in this priority queue than in the cohort overall.

Expanding-window temporal validation produces a mean ROC-AUC of approximately **0.89**. Churn prevalence also changes materially between cohorts, making cohort-level monitoring an important part of the production design.

---

## A Critical Analytical Decision: What Is “Day 14”?

Administrative enrollment is not the same as the actual start of learning.

The median gap between `Дата зачисления` and the first scheduled lesson is approximately **26 days**. Using enrollment as the observation-window anchor would therefore classify many students as inactive before their course had actually started.

The D14 clock is instead anchored to the scheduled start of the student's learning cohort:

```text
stats.id параллели
        ↕
groups.group_template_id
        ↓
min(groups.starts_at) = learning_start
```

Only events satisfying the following condition are eligible for model features:

```text
learning_start <= event_timestamp <= learning_start + 14 days
```

This point-in-time design prevents future information from entering the prediction window.

---

## Data Leakage Prevention

The model is designed as a true early-warning system rather than a reconstruction of final student status.

Features based on final module outcomes or whole-module aggregates are excluded from the predictor set. In particular, final progress, final assessment outcomes, and other information that becomes available at or after churn are not used as D14 predictors.

The current reproducible feature set is built from timestamped behavior available by Day 14, including:

- number of media sessions;
- active learning days;
- unique learning resources;
- watch depth;
- week-1 and week-2 activity;
- activity during the last three days of the observation window;
- recency;
- inactivity gaps;
- intervals between learning sessions.

---

## Early Behavioral Signals

Students who later churn show substantially weaker learning cadence during the first 14 days.

| D14 behavior | Retained | Future churn |
|---|---:|---:|
| Media sessions | **9.58** | **1.69** |
| Active media days | **2.82** | **0.60** |
| Unique learning resources | **6.83** | **1.14** |
| Average watch depth | **46.8%** | **10.3%** |
| Recency at D14 | **4.43 days** | **11.22 days** |
| Maximum inactivity gap | **8.01 days** | **12.42 days** |

The main pattern is not a single action. It is a **loss of learning cadence**: fewer active days, narrower content exposure, weaker continuation into week 2, and longer inactivity by D14.

These are predictive associations, not causal effects.

---

## Model Comparison

| Model | ROC-AUC | PR-AUC | Brier | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.911** | **0.800** | **0.118** | 0.805 | 0.527 | 2.63× |
| **Random Forest** | 0.906 | 0.797 | 0.122 | **0.821** | **0.537** | **2.69×** |
| CatBoost | 0.898 | 0.778 | 0.149 | 0.813 | 0.532 | 2.66× |
| Gradient Boosting | 0.860 | 0.722 | 0.140 | 0.748 | 0.489 | 2.45× |
| HistGradientBoosting | 0.870 | 0.729 | 0.149 | 0.675 | 0.441 | 2.21× |

Logistic Regression performs best on overall ROC-AUC, PR-AUC, and calibration. However, **Random Forest is selected as the operating model** because the product has a concrete capacity constraint: the retention team can manually contact only a limited share of students.

At the Top-20% operating point, Random Forest provides the strongest Precision, Recall, and Lift.

This is an intentional product decision rather than simply selecting the most complex algorithm.

---

## Capacity-Aware Decision Making

The optimal risk threshold depends on the retention team's available capacity.

| Contact capacity | Precision | Recall |
|---|---:|---:|
| Top 10% | 90.3% | 29.8% |
| Top 20% | 82.1% | 53.7% |
| Top 30% | 69.7% | 68.6% |

Increasing the intervention queue improves recall but reduces precision.

This makes `Precision@K`, `Recall@K`, and `Lift@K` more useful operational metrics than a generic probability threshold of 0.5.

---

## Product Activation

The model is designed to support a prioritized retention workflow.

| Risk segment | Recommended action |
|---|---|
| 🔴 High | Personal curator outreach within 24 hours |
| 🟠 Medium | Automated nudge and study-plan recommendation |
| 🟢 Low | Standard learning journey |

The model answers **who should receive attention first**.

Behavioral diagnostics such as recency, week-2 continuation, and inactivity gaps help the retention team understand **why a student was prioritized** and what type of intervention may be appropriate.

---

## Experimentation Framework

Prediction alone does not prove that an intervention reduces churn.

A retention pilot should randomize eligible high-risk students into:

- **Control:** business-as-usual;
- **Treatment:** standardized curator outreach.

### Primary metric

**Module 1 → Module 2 Transition Rate**

### Secondary metric

**Course Completion Rate**

### Guardrails

- curator workload;
- intervention cost;
- complaints and opt-outs.

The causal business effect is:

```text
Incremental retained students
=
Eligible contacted students
×
Measured intervention uplift
```

The model determines **who to prioritize**. The experiment determines **whether the intervention works**.

---

## Metrics Framework

| Level | Metric |
|---|---|
| North Star | Course Completion Rate |
| Leading retention metric | Module 1 → Module 2 Transition Rate |
| ML ranking | PR-AUC |
| Operational targeting | Precision@Top-K, Recall@Top-K, Lift@Top-K |
| Behavioral diagnostics | Active days, recency, inactivity gap, week-2 continuation |
| Experiment | Incremental transition-rate uplift |

---

## Public Data Architecture

Raw student-level data is intentionally excluded from the public repository.

The portfolio is split into a private modeling layer and a public presentation layer:

```text
PRIVATE RAW DATA
stats / groups / media logs
        ↓
Notebook + feature pipeline
        ↓
PUBLIC-SAFE AGGREGATES
results/*.csv
        ↓
Streamlit dashboard
```

The public Streamlit application reads only aggregated outputs from `results/`. It does **not** require private student-level CSV files.

### Public result files

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

### Full local retraining

To reproduce the model from raw data, place the private source files in:

```text
data/
├── stats__module_1.csv
├── groups.csv
└── wk_media_view_sessions.csv
```

These files are excluded from Git through `.gitignore`.

---

## Repository Structure

```text
cifrium-retention-intelligence/
├── README.md
├── app.py
├── requirements.txt
├── data/
│   └── README.md
├── docs/
│   ├── BUSINESS_FINDINGS.md
│   ├── MODEL_REPORT.md
│   └── EXPERIMENT_DESIGN.md
├── notebooks/
│   └── 01_cifrium_early_churn_product_case.ipynb
├── results/
│   ├── model_metrics.csv
│   ├── temporal_validation.csv
│   ├── behavioral_summary.csv
│   ├── feature_importance.csv
│   ├── funnel_summary.csv
│   ├── capacity_curve.csv
│   └── cohort_performance.csv
└── src/
    └── features.py
```

---

## Run the Project

### Streamlit dashboard

The public dashboard requires only the files committed to `results/`.

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Full analytical pipeline

Full retraining additionally requires the private raw data in `data/`.

```bash
jupyter notebook notebooks/01_cifrium_early_churn_product_case.ipynb
```

---

## Project Takeaway

This project treats churn prediction as a **product decision system**, not only as a classification problem.

The workflow is:

**Measure → Predict → Prioritize → Intervene → Experiment → Learn**

The final objective is not a high ROC-AUC. It is to improve student retention and ultimately increase **Course Completion Rate**.
