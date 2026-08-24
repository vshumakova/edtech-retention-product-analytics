# Portfolio Case Summary

## Problem
36% of students are lost between Module 1 and Module 2.

## Product goal
Identify churn risk at Day 14 of actual learning so the retention team can intervene before the main drop-off.

## North Star
Course Completion Rate.

## Selected decision system
Regularized Logistic Regression using point-in-time-safe D14 media + training behavior.

## Future-cohort result
- ROC-AUC: 0.917
- PR-AUC: 0.815
- Precision@Top-20%: 85.4%
- Recall@Top-20%: 55.9%
- Lift@Top-20%: 2.79×

## Why this is a product analytics case
The project links:
retention diagnosis → prediction timing → leakage-safe feature design → capacity-aware ranking → intervention workflow → A/B test → North Star impact.

## Main product insight
Early churn is associated with a loss of learning cadence: low activity, weak week-2 continuation, long inactivity, and limited training engagement.

## Business recommendation
Use the model as a prioritization layer for a controlled curator-outreach pilot. Measure causal uplift on M1 → M2 Transition Rate before scaling.
