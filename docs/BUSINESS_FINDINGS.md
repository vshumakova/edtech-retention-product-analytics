# Business Findings

## Executive summary

Cifrium's retention challenge is concentrated early in the learning journey: **36% of students are lost between Modules 1 and 2**.

This makes Module 1 the highest-value intervention zone. Course Completion Rate remains the North Star, while **M1 → M2 Transition Rate** is the faster leading outcome for evaluating early retention work.

The proposed operating model is to calculate a **Churn Risk Score on Day 14**, prioritize students under limited curator capacity, and select an intervention based on the student's early behavioral pattern.

> The objective is not to predict churn for reporting purposes. The objective is to create enough lead time for an intervention.

---

## 1. Product opportunity

### North Star

**Course Completion Rate**

Share of enrolled students who complete the full course and pass the final assessment in Module 4.

### Decision metric

**Module 1 → Module 2 Transition Rate**

This metric is closer to the intervention point and can respond faster than final course completion.

### Decision moment

**Day 14 after enrollment**

By this point, the platform has accumulated early behavioral information while the student's final Module 1 outcome is still in the future.

---

## 2. Early-warning framework

The analytical snapshot focuses on behaviors that can plausibly indicate disengagement:

- number of active learning days;
- inactivity and recency;
- LMS session frequency;
- lesson breadth;
- task attempts and task engagement when timestamped;
- media consumption when timestamped;
- delay between enrollment and first activity.

The project compares these signals between students who later churn and students who remain.

### Interpretation principle

Predictive signals are useful only if they can support a decision.

For example:

| Signal | Possible friction | Product response |
|---|---|---|
| Zero / very low early activity | onboarding, access or motivation | curator outreach |
| Long inactivity gap | disengagement | reactivation workflow |
| Low task engagement | academic difficulty / avoidance | study plan or academic support |
| Low content consumption | content-format friction | alternative learning path |
| High model risk without obvious inactivity | more complex risk profile | curator diagnosis |

The model ranks **priority**. Behavioral analytics supports the **intervention choice**.

---

## 3. Recommended operating model

### High risk

Personal curator outreach within 24 hours.

Goal: identify the blocker and choose a relevant intervention.

### Medium risk

Lower-cost intervention such as an automated nudge, study-plan reminder or content recommendation.

### Low risk

Standard learning journey without costly manual intervention.

Risk thresholds should be determined by available curator capacity and measured intervention economics rather than by an arbitrary probability threshold.

---

## 4. Metrics hierarchy

### North Star

**Course Completion Rate**

### Retention metrics

- M1 → M2 Transition Rate;
- Module 1 retention;
- cohort retention.

### Leading behavioral metrics

- D14 active-student share;
- zero-activity share;
- median recency;
- task engagement;
- content engagement.

### Decision-system metrics

- share of cohort classified high risk;
- Precision@Top-K;
- Recall@Top-K;
- Lift@Top-K;
- intervention coverage.

### Experiment metrics

- incremental M1 → M2 transition;
- incremental Course Completion Rate;
- intervention cost;
- curator workload;
- complaints / opt-outs.

---

## 5. Business impact logic

Model performance should not be translated directly into revenue or retention impact.

The expected value depends on both targeting and the effectiveness of the intervention:

```text
Incremental retained students
=
Students contacted
× Precision among contacted students
× Causal intervention uplift
```

Expected downstream course completions can then be estimated from the probability that an additionally retained student ultimately completes the course.

This decomposition prevents the predictive model from being credited for impact that actually depends on the intervention.

---

## 6. Recommended next steps

1. Run the D14 scoring pipeline on matured historical cohorts.
2. Select an operational Top-K threshold from curator capacity.
3. Attach behavioral diagnostics to the high-risk queue.
4. Launch one clearly defined intervention for high-risk students.
5. Randomize eligible students into treatment and control.
6. Measure incremental M1 → M2 transition.
7. Track downstream Course Completion Rate.
8. Scale the workflow only after causal uplift and operational economics are confirmed.

---

## Decision summary

| Question | Recommendation |
|---|---|
| Where should Cifrium intervene? | During Module 1 |
| When should risk be scored? | Day 14 |
| Who should be prioritized? | Highest-risk students within curator capacity |
| What should the model optimize operationally? | Risk concentration / Recall@Top-K |
| How should impact be proven? | Randomized retention experiment |
| What remains the North Star? | Course Completion Rate |

---

## Evidence status

The **36% M1 → M2 loss** and the project business context are part of the supplied case framing.

Final D14 behavioral effect sizes, selected-model metrics and measured intervention uplift must be populated only after the corresponding pipeline or experiment has been run. This document intentionally does not present illustrative dashboard values as observed business results.
