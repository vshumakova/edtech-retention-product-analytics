# 🎓 Cifrium Early Churn Intelligence

### Product Analytics & Machine Learning for proactive student retention

> An end-to-end product analytics case: from retention diagnosis and early behavioral signals to a Day-14 churn risk model, intervention prioritization, and experiment design.

<p align="center">
  <b>Product Analytics</b> · <b>Retention</b> · <b>Machine Learning</b> · <b>Experimentation</b> · <b>Streamlit</b>
</p>

---

## TL;DR

Cifrium is an online education platform with a four-module course. The main retention bottleneck occurs early in the student journey: **36% of students are lost between Modules 1 and 2**.

The goal of this project is to identify students at elevated churn risk **on Day 14 after enrollment**, while there is still time to intervene.

The solution combines:

- product funnel and cohort analysis;
- point-in-time behavioral feature engineering;
- churn prediction with time-based validation;
- capacity-aware ML metrics such as `Recall@Top-K` and `Lift@Top-K`;
- an operational risk segmentation for curators;
- an A/B-test framework to measure causal retention uplift;
- an interactive Streamlit dashboard for product and operations teams.

**North Star Metric:** `Course Completion Rate`  
**Decision Metric:** `Module 1 → Module 2 Transition Rate`  
**ML Output:** `D14 Churn Risk Score`

---

## 🎯 Product problem

Course completion is a lagging metric: by the time a student has already dropped out, the opportunity to influence the outcome is gone.

The product question is therefore:

> **Can we detect early disengagement by Day 14 and prioritize students for intervention before they leave the course?**

The project treats ML as one component of a broader retention system:

```text
Early learning behavior
        ↓
D14 Churn Risk Score
        ↓
Prioritized intervention
        ↓
Higher M1 → M2 transition
        ↓
Higher Course Completion Rate
```

The model does not automatically make decisions about students. It ranks intervention priority so that limited curator capacity can be allocated where it is most useful.

---

## 📊 Product analytics

### 1. Retention funnel

The first step is identifying where the largest loss occurs across the four-module journey.

The analysis focuses on:

- student volume at each module;
- module completion and churn rates;
- transition rates between modules;
- retention dynamics across enrollment cohorts.

The largest opportunity is the transition from **Module 1 to Module 2**, making the first module the primary intervention window.

### 2. Cohort analysis

Retention is tracked by enrollment cohort rather than relying only on an overall average.

This helps identify:

- changes in retention over time;
- deterioration in newer cohorts;
- possible differences in acquisition or teaching periods;
- whether model performance remains stable for future cohorts.

### 3. Early behavioral analytics

For every student, the project reconstructs their learning behavior during the first 14 days.

Examples of D14 signals:

| Feature family | Examples |
|---|---|
| **Activity** | active days, LMS events, sessions |
| **Recency** | days since last activity |
| **Cadence** | inactivity gaps, events per active day |
| **Learning breadth** | unique lessons visited |
| **Tasks** | attempts, unique tasks, solve rate |
| **Media** | watch depth, media sessions |
| **Timing** | delay to first activity, weekend/evening activity |

Behavioral differences between retained and churned students are evaluated using standardized effect sizes and empirical risk curves.

This provides not only predictive signals, but also **potential intervention levers**.

---

## ⏱ Point-in-time analytical design

The model represents the student exactly as they could have been observed at the scoring moment.

```text
Enrollment                         Outcome
    │                                 │
    ├──────── Observation ────────┤   │
    Day 0                         Day 14
                                  ↑
                              Risk score
```

The feature cutoff is:

```python
event_timestamp <= enrollment_timestamp + 14 days
```

Only events available by that moment are eligible for feature engineering.

Final assessment results, final module status, and whole-period aggregates are not used as predictors.

This makes the offline evaluation representative of the intended production decision.

---

## 🤖 Machine learning

The modeling stage compares several approaches:

### Logistic Regression

Used as an interpretable baseline and calibration reference.

### HistGradientBoosting

Captures non-linear relationships between behavioral signals using a standard scikit-learn pipeline.

### CatBoost

Used as the stronger tabular model for non-linear interactions and categorical context.

### Validation strategy

Students are split **chronologically by enrollment date**:

```text
Earlier cohorts                     Later cohorts
┌────────────────────────────┐      ┌──────────────┐
│           TRAIN            │      │     TEST     │
└────────────────────────────┘      └──────────────┘
```

This approximates the real production setting: train on historical students and predict risk for future cohorts.

---

## 📈 Model evaluation

A churn model is useful only if it improves prioritization under real operational constraints.

The project therefore evaluates both statistical and product-oriented metrics.

| Metric | Why it matters |
|---|---|
| **ROC-AUC** | Overall ranking quality |
| **PR-AUC** | Performance on the positive churn class |
| **Brier Score** | Probability quality / calibration |
| **Precision@Top-K** | Churn concentration among contacted students |
| **Recall@Top-K** | Share of future churners captured within capacity |
| **Lift@Top-K** | Improvement over random targeting |

### Why Top-K matters

Suppose curators can personally contact only **20% of a cohort**.

The operational question is not:

> “Is predicted probability above 0.5?”

It is:

> **“If we contact the 20% highest-risk students, how many future churners will we reach?”**

This directly connects model evaluation to the product workflow.

---

## 🧭 From prediction to intervention

The score is converted into a prioritized operations queue.

| Risk segment | Product action |
|---|---|
| 🔴 **High** | Personal curator outreach within 24 hours |
| 🟠 **Medium** | Automated nudge + study-plan recommendation |
| 🟢 **Low** | Standard learning journey |

The behavioral profile helps determine *why* the student may be at risk.

For example:

```text
Long inactivity gap
→ reactivation outreach

Low task engagement
→ academic support / study plan

Low content consumption
→ alternative learning path

Zero early activity
→ onboarding or access check
```

The model determines **who should receive attention first**.  
Behavioral analytics helps determine **what intervention may be appropriate**.

---

## 🧪 Measuring business impact

Predictive performance alone does not demonstrate that the system improves retention.

The intervention should therefore be evaluated experimentally.

### Proposed experiment

**Population:** students classified as high risk on D14  
**Treatment:** targeted curator intervention  
**Control:** business-as-usual  

**Primary metric**

`Module 1 → Module 2 Transition Rate`

**Secondary metric**

`Course Completion Rate`

**Guardrails**

- curator workload;
- intervention cost;
- complaints / opt-outs.

The expected impact can be decomposed as:

```text
Incremental retained students
=
Students contacted
× Precision among contacted students
× Causal intervention uplift
```

This deliberately separates:

1. **model quality** — can we identify risk?
2. **targeting capacity** — how many students can we reach?
3. **causal effectiveness** — does the intervention actually change behavior?

---

## 🖥 Interactive dashboard

The repository includes a Streamlit application that turns the analysis into a product decision tool.

### Executive Overview

Retention funnel, key KPIs, risk mix, and the overall operating logic.

### Retention Analytics

Cohort trends and behavioral risk curves for early learning signals.

### Risk Operations

A ranked student queue containing:

- D14 risk score;
- risk segment;
- early behavioral indicators;
- recommended intervention.

### Model Quality

Model metrics, risk concentration, lift by decile, and the strongest D14 signals.

### Impact Simulator

A scenario tool connecting:

```text
model precision
+ curator capacity
+ intervention uplift
→ expected retained students
→ expected Course Completion impact
```

Run locally:

```bash
streamlit run app.py
```

---

## 🗂 Repository structure

```text
cifrium-early-churn/
│
├── README.md
├── app.py
├── requirements.txt
│
├── notebooks/
│   └── 01_cifrium_early_churn_product_case.ipynb
│
├── src/
│   ├── features.py
│   ├── evaluation.py
│   └── dashboard.py
│
├── data/
│   └── README.md
│
└── .streamlit/
    └── config.toml
```

### Why both `.ipynb` and `.py`?

The repository separates analytical exploration from reusable code.

**Jupyter Notebook**

`notebooks/01_cifrium_early_churn_product_case.ipynb`

Contains the end-to-end analytical narrative:

- product framing;
- funnel analysis;
- cohort analysis;
- behavioral EDA;
- feature validation;
- model comparison;
- product interpretation;
- experiment design.

**Python modules**

`src/`

Contain reusable logic for:

- point-in-time feature engineering;
- ML evaluation;
- dashboard data preparation.

**Streamlit**

`app.py`

Provides the stakeholder-facing decision layer.

---

## 🚀 How to run

### 1. Clone the repository

```bash
git clone <repository-url>
cd cifrium-early-churn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analytical notebook

```bash
jupyter notebook notebooks/01_cifrium_early_churn_product_case.ipynb
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

---

## 🔐 Data privacy

Student-level source data is not included in the public repository.

The Streamlit application supports a **portfolio demo mode** with illustrative data so that the dashboard can be reviewed without exposing private educational records.

Prepared analytical outputs can be connected locally through the `data/` directory.

---

## 💡 Key product takeaways

**1. Retention should be attacked where there is still time to act.**  
Course Completion Rate is the North Star, but the first module provides the earliest high-value intervention opportunity.

**2. Prediction and intervention are different problems.**  
The model identifies students at risk; only an experiment can establish whether an intervention reduces churn.

**3. Model evaluation should reflect operational capacity.**  
For a curator team with limited capacity, Recall@Top-K and Lift@Top-K are more actionable than accuracy.

**4. Behavioral signals should translate into actions.**  
Risk scoring becomes more useful when inactivity, weak task engagement, or low content consumption map to different intervention playbooks.

**5. The final product is a decision system, not a model file.**  
The value comes from the complete loop:

```text
Measure → Predict → Prioritize → Intervene → Experiment → Learn
```

---

## 🛠 Tech stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `CatBoost` · `Matplotlib` · `Plotly` · `Streamlit` · `Jupyter`

---

### Project focus

**Product Analytics · Retention Analytics · Behavioral Analytics · Machine Learning · Experiment Design · Data Product**
