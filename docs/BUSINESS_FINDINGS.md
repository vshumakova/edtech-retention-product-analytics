# Business Findings

## Executive Summary

Cifrium loses **36% of students between Module 1 and Module 2**, making early learning behavior the highest-value retention opportunity.

The extended D14 model combines **media engagement and training behavior**. On a future-cohort holdout it achieves:

- ROC-AUC: **0.917**
- PR-AUC: **0.815**
- Precision@Top-20%: **85.4%**
- Recall@Top-20%: **55.9%**
- Lift@Top-20%: **2.79×**

This means that a retention team working with only 20% of the cohort can reach more than half of all future churners.

## What Changed After Expanding the Feature Store

Adding point-in-time-safe training behavior improves the operational queue:

| Feature set | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|
| Media only | 80.5% | 52.7% | 2.63× |
| **Media + training** | **85.4%** | **55.9%** | **2.79×** |

Course/account and access timing were also tested, but they did not improve the operational Top-20% metrics. They are therefore excluded from the selected model.

This is important from a product perspective: **feature selection is based on decision value, not on the number of available tables.**

## Main Behavioral Pattern

Future churners show a broader loss of learning cadence:

- less media engagement;
- fewer active learning days;
- weaker week-2 continuation;
- longer recency;
- fewer training starts;
- fewer safely completed training activities by D14.

The recommended intervention workflow remains:

1. score every eligible student at D14;
2. prioritize the highest-risk segment;
3. attach behavioral diagnostics;
4. test a standardized intervention through randomization;
5. measure M1 → M2 transition uplift;
6. verify downstream Course Completion impact.

## Product Recommendation

Use the model as a **curator prioritization layer**, not as an automated decision maker.

The next business milestone is not another offline model improvement. It is a controlled retention experiment that measures whether targeted outreach creates incremental retention.
