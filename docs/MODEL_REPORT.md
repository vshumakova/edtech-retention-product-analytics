# Model Report

## Prediction objective

Estimate future Module-1 churn risk using only information available by the end of the first **14 learning days**.

| Component | Definition |
|---|---|
| Unit of analysis | Student at D14 |
| Learning-start anchor | First scheduled lesson of the student's parallel |
| Observation window | Learning start → learning start + 14 days |
| Target | `1 = Отчислен`, `0 = Завершил` |
| Validation | Chronological future-cohort holdout |
| Operational constraint | Curators can contact only a fraction of the cohort |

The final status is used only as the matured target.

---

## Why D14 is anchored to learning start

The raw data shows that administrative enrollment and actual learning start are different events.

For students with a valid schedule, the median gap between `Дата зачисления` and the first scheduled lesson is about **26 days**.

The learning start is reconstructed as:

```text
stats.id параллели
        ↕
groups.group_template_id
        ↓
min(groups.starts_at)
```

This mapping provides a valid learning start for **2,972 students**.

Using enrollment date as the D14 anchor would incorrectly classify many students as inactive before their course had started.

---

## Point-in-time feature policy

Every behavioral event must satisfy:

```text
learning_start <= event_timestamp <= learning_start + 14 days
```

The model uses safe schedule context and timestamped D14 media behavior.

Examples:

- media sessions;
- active media days;
- unique resources;
- watch depth;
- week-1 / week-2 activity;
- last-three-day activity;
- recency;
- inactivity gaps;
- time between sessions.

Final module progress, assessment outcomes, final status, and whole-period aggregates are excluded from predictors.

---

## Holdout design

The dataset is ordered by scheduled Module-1 start.

- **Train:** 2,357 students
- **Future holdout:** 615 students
- **Train churn rate:** 35.2%
- **Holdout churn rate:** 30.6%

The headline result does not use a random row split.

---

## Model comparison

| Model | ROC-AUC | PR-AUC | Brier | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.911** | **0.800** | **0.118** | 0.805 | 0.527 | 2.63× |
| **Random Forest** | 0.906 | 0.797 | 0.122 | **0.821** | **0.537** | **2.69×** |
| CatBoost | 0.898 | 0.778 | 0.149 | 0.813 | 0.532 | 2.66× |
| Gradient Boosting | 0.860 | 0.722 | 0.140 | 0.748 | 0.489 | 2.45× |
| HistGradientBoosting | 0.870 | 0.729 | 0.149 | 0.675 | 0.441 | 2.21× |

### Selected operating model: Random Forest

Logistic Regression is the strongest general offline model by ROC-AUC, PR-AUC and Brier score.

However, the product has an explicit capacity constraint: only the highest-risk share of the cohort can receive manual intervention. **Random Forest is therefore selected for the operating queue because it provides the strongest Precision@20%, Recall@20% and Lift@20%.**

This is an intentional product trade-off rather than choosing the most complex algorithm.

---

## Capacity curve

For the future holdout:

| Contact capacity | Precision | Recall |
|---|---:|---:|
| Top 10% | 90.3% | 29.8% |
| Top 20% | 82.1% | 53.7% |
| Top 30% | 69.7% | 68.6% |

At 20% capacity, the queue captures more than half of future churners.

The threshold should therefore be chosen from curator capacity and intervention economics, not from a generic probability cutoff of 0.5.

---

## Temporal robustness

Three expanding-window temporal checks for the selected Random Forest:

| Fold | Test churn | ROC-AUC | PR-AUC | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 29.7% | 0.849 | 0.632 | 0.650 | 0.437 | 2.18× |
| 2 | 65.3% | 0.913 | 0.947 | 0.990 | 0.306 | 1.52× |
| 3 | 30.6% | 0.906 | 0.797 | 0.821 | 0.537 | 2.69× |

Mean ROC-AUC ≈ **0.890**.

Target prevalence changes sharply between cohorts. Model monitoring must therefore be cohort-aware.

---

## Behavioral findings

Across the modeling sample:

| D14 behavior | Retained | Churn |
|---|---:|---:|
| Media sessions | 9.58 | 1.69 |
| Active media days | 2.82 | 0.60 |
| Unique media resources | 6.83 | 1.14 |
| Average watch depth | 46.8% | 10.3% |
| Share watched ≥80% | 32.2% | 5.2% |
| Recency at D14 | 4.43 days | 11.22 days |
| Maximum inactivity gap | 8.01 days | 12.42 days |
| Week-2 sessions | 5.94 | 0.80 |
| Sessions in last 3 days | 3.32 | 0.43 |

The main pattern is loss of learning cadence rather than one isolated behavior.

---

## Calibration

The selected Random Forest has a holdout Brier score of **0.122**.

Logistic Regression is slightly better calibrated offline at **0.118**.

For initial deployment, the model score is best treated as a **ranking score**. Probability calibration should be monitored before presenting literal churn probabilities to operations.

---

## Limitations

1. The current reproducible model uses timestamped media behavior because it has a direct student identifier and event time.
2. Timestamped task behavior could improve prediction once it is joined through a scalable point-in-time mapping.
3. The evaluated cohorts are relatively close in calendar time; longer out-of-time validation is desirable.
4. Churn prevalence varies materially across cohorts.
5. Feature importance and behavioral differences are predictive associations, not causal drivers.
6. Model quality does not prove that an intervention reduces churn.

---

## Production monitoring

Monitor:

- cohort churn rate;
- ROC-AUC / PR-AUC;
- Precision@Top-K;
- Recall@Top-K;
- Lift@Top-K;
- calibration;
- risk-score distribution;
- high-risk population share;
- feature drift;
- intervention coverage;
- experiment uplift.

Aggregated public-safe outputs are stored in `results/`. Raw student-level data is excluded from the repository.
