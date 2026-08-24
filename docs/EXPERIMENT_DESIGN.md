# Retention Experiment Design

## Goal

Measure whether a targeted intervention for students identified as high risk on Day 14 **causally improves retention**.

The predictive model answers:

> Who is at elevated risk?

The experiment answers:

> Does our intervention change the outcome?

These questions must be evaluated separately.

---

## Hypothesis

> A targeted Day-14 intervention for high-risk students increases the probability of transitioning from Module 1 to Module 2 compared with business-as-usual.

---

## Experiment population

Students who:

1. reach the D14 scoring point;
2. satisfy operational eligibility criteria;
3. fall inside the selected high-risk segment.

The exact risk threshold should reflect available curator capacity.

---

## Randomization

Eligible high-risk students are randomly assigned to:

### Control

Business-as-usual learning journey.

### Treatment

Defined retention intervention.

For the first experiment, the treatment should be intentionally simple and standardized enough to measure.

Example:

**Personal curator outreach within 24 hours of D14 scoring.**

The purpose is to diagnose the student's blocker and help them return to an active study plan.

---

## Metrics

### Primary metric

**M1 → M2 Transition Rate**

Chosen because it is close to the intervention and directly reflects the main retention bottleneck.

### Secondary metric

**Course Completion Rate**

Measures whether the early retention gain survives through the complete course journey.

### Diagnostic metrics

- post-intervention active days;
- return-to-learning rate;
- task engagement;
- time to next learning event.

### Guardrails

- curator workload;
- intervention cost;
- complaint / opt-out rate;
- negative engagement signals.

---

## Success criterion

The primary decision should be based on the **difference in M1 → M2 transition between treatment and control**, together with uncertainty around that estimate.

Do not use an improvement in model score or engagement alone as evidence of retention impact.

---

## Analysis framework

Report:

```text
Control transition rate
Treatment transition rate
Absolute uplift
Relative uplift
Confidence interval
Sample size
```

Also inspect downstream Course Completion Rate once the cohort has matured.

---

## Why randomize only eligible high-risk students?

The intended product is a targeted retention workflow.

Randomizing within the operationally eligible population estimates the causal effect of the intervention **where the system would actually use it**.

A broader experiment may be useful later to test whether risk-based targeting itself creates incremental value relative to untargeted outreach.

---

## From experiment to business impact

After causal uplift is measured:

```text
Incremental retained students
=
Eligible students contacted
× measured absolute intervention uplift
```

For planning scenarios, model Precision@Top-K can help estimate how concentrated underlying risk is in the contacted group, but the experiment remains the source of causal impact.

---

## Follow-up experiments

Once the first intervention is validated, test:

1. personal curator outreach vs automated nudge;
2. intervention timing: D7 vs D14;
3. different treatments by behavioral risk reason;
4. different capacity thresholds;
5. risk-based targeting vs broader outreach.

This evolves the project from a churn model into an experimentation-driven retention system.
