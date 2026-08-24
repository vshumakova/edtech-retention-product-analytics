# Model Report

## Objective

Predict future Module-1 churn at the end of the first 14 learning days using only point-in-time-safe information.

## Selected Feature Store

The operating model uses:

- schedule / cohort context from `stats__module_1.csv` and `groups.csv`;
- D14 media behavior from `wk_media_view_sessions.csv`;
- D14 training behavior from `user_trainings.csv`;
- static training metadata from `trainings.csv`.

Course-account and access timing sources were audited but rejected after ablation because they did not improve Top-20% targeting.

## Training Leakage Policy

A training can start before D14 and finish after D14. Therefore final values cannot be used merely because `started_at` is inside the observation window.

Rules:

```text
started_at <= cutoff
→ count that the training was started

finished_at <= cutoff
→ solved_tasks_count / submitted_answers_count / earned_points are eligible

mark_saved_at <= cutoff
→ mark is eligible
```

## Source Ablation

| Feature set | ROC-AUC | PR-AUC | Brier | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|
| Media only | 0.913 | 0.806 | 0.117 | 0.805 | 0.527 | 2.63× |
| **Media + training** | **0.917** | **0.815** | 0.113 | **0.854** | **0.559** | **2.79×** |
| Media + course/access | 0.911 | 0.799 | 0.115 | 0.805 | 0.527 | 2.63× |
| All audited sources | 0.918 | 0.817 | **0.111** | 0.805 | 0.527 | 2.63× |

### Selection decision

The “all audited sources” version is marginally better on global ROC-AUC / PR-AUC / Brier, but the **media + training model is materially better for the actual business constraint: Top-20% intervention capacity**.

For this reason, the operating model is **regularized Logistic Regression with media + training features**.

## Future-Cohort Performance

- ROC-AUC: **0.917**
- PR-AUC: **0.815**
- Brier: **0.113**
- Precision@20%: **85.4%**
- Recall@20%: **55.9%**
- Lift@20%: **2.79×**

## Temporal Validation

| Fold | Test churn | ROC-AUC | PR-AUC | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 18.6% | 0.848 | 0.503 | 50.4% | 54.2% | 2.71× |
| 2 | 67.1% | 0.919 | 0.954 | 99.2% | 29.7% | 1.48× |
| 3 | 30.6% | 0.917 | 0.815 | 85.4% | 55.9% | 2.79× |

The shift in churn prevalence is substantial. Performance monitoring must be cohort-aware.

## Limitations

1. Training behavior covers a smaller share of students than media behavior, so missingness itself is informative.
2. `user_activity_histories` and `user_answers` could contain additional signal, but they require a heavier scalable event pipeline.
3. The current cohort time range is short; longer out-of-time validation is still needed.
4. Predictive features are not causal drivers.
5. The model's business value depends on intervention uplift.

## Production Use

Treat the score primarily as a **ranking score**. Select the operating threshold from curator capacity and intervention economics.
