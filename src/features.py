from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_id(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def build_parallel_start(groups: pd.DataFrame) -> pd.DataFrame:
    """
    Map stats 'id параллели' to the first scheduled lesson of that parallel.

    groups.group_template_id corresponds to the parallel identifier.
    The minimum scheduled starts_at is used as the learning-start anchor.
    """
    g = groups.copy()
    g["parallel_id"] = pd.to_numeric(
        normalize_id(g["group_template_id"]), errors="coerce"
    )
    g["starts_at"] = pd.to_datetime(g["starts_at"], errors="coerce")

    return (
        g.dropna(subset=["parallel_id", "starts_at"])
        .groupby("parallel_id", as_index=False)
        .agg(
            learning_start=("starts_at", "min"),
            scheduled_lessons=("starts_at", "size"),
        )
    )


def build_student_base(
    stats_m1: pd.DataFrame,
    groups: pd.DataFrame,
    observation_days: int = 14,
) -> pd.DataFrame:
    """
    Create one row per student and attach the scheduled Module-1 learning start.

    Final status is retained only as the target. Final progress / assessment
    columns never enter the feature matrix.
    """
    s = stats_m1[stats_m1["user_id"].notna()].copy()
    s["user_id"] = normalize_id(s["user_id"])
    s["parallel_id"] = pd.to_numeric(s["id параллели"], errors="coerce")
    s["enrollment_date"] = pd.to_datetime(s["Дата зачисления"], errors="coerce")
    s["churn"] = s["Статус"].map({"Отчислен": 1, "Завершил": 0})

    keep = ["user_id", "parallel_id", "enrollment_date", "churn", "Уровень", "course_id"]
    s = s.sort_values("user_id").drop_duplicates("user_id", keep="first")[keep]

    starts = build_parallel_start(groups)
    s = s.merge(starts, on="parallel_id", how="left")
    s = s.dropna(subset=["learning_start", "churn"]).copy()
    s["churn"] = s["churn"].astype(int)
    s["cutoff_d14"] = s["learning_start"] + pd.Timedelta(days=observation_days)
    s["enrollment_lead_days"] = (
        s["learning_start"] - s["enrollment_date"]
    ).dt.total_seconds() / 86400

    # Safe cohort size: students already enrolled by the learning start.
    cohort_sizes = []
    for pid, g in s.groupby("parallel_id"):
        start = g["learning_start"].iloc[0]
        cohort_sizes.append(
            (pid, int((g["enrollment_date"] <= start).sum()))
        )
    cohort_sizes = pd.DataFrame(
        cohort_sizes, columns=["parallel_id", "cohort_size_at_start"]
    )
    s = s.merge(cohort_sizes, on="parallel_id", how="left")
    s["start_weekday"] = s["learning_start"].dt.dayofweek.astype("string")
    s["start_hour"] = s["learning_start"].dt.hour

    return s


def build_d14_media_features(
    base: pd.DataFrame,
    media: pd.DataFrame,
    observation_days: int = 14,
) -> pd.DataFrame:
    """
    Aggregate media behavior using events between learning_start and D14 only.
    """
    m = media.copy()
    m["user_id"] = normalize_id(m["viewer_id"])
    m["started_at"] = pd.to_datetime(m["started_at"], errors="coerce")
    m = m.dropna(subset=["started_at"])

    d = m.merge(
        base[["user_id", "learning_start", "cutoff_d14"]],
        on="user_id",
        how="inner",
        validate="many_to_one",
    )
    d = d[
        (d["started_at"] >= d["learning_start"])
        & (d["started_at"] <= d["cutoff_d14"])
    ].copy()

    if len(d) and not (d["started_at"] <= d["cutoff_d14"]).all():
        raise AssertionError("Future media event leaked into D14 features.")

    d["date"] = d["started_at"].dt.normalize()
    total = pd.to_numeric(d["segments_total"], errors="coerce").replace(0, np.nan)
    viewed = pd.to_numeric(d["viewed_segments_count"], errors="coerce")
    d["watch_pct"] = (viewed / total).clip(0, 1)
    d["watched80"] = (d["watch_pct"] >= 0.8).astype(int)
    d["watched50"] = (d["watch_pct"] >= 0.5).astype(int)
    d["is_live"] = d["kind"].eq("ulms_live").astype(int)

    rows = []
    for uid, g in d.groupby("user_id", sort=False):
        g = g.sort_values("started_at").copy()
        start = g["learning_start"].iloc[0]
        cutoff = g["cutoff_d14"].iloc[0]

        g["day_idx"] = (g["started_at"] - start).dt.total_seconds() / 86400
        w1 = g[g["day_idx"] < 7]
        w2 = g[g["day_idx"] >= 7]
        last3 = g[g["day_idx"] >= 11]

        dates = g["date"].drop_duplicates().sort_values()
        offsets = (dates - start.normalize()).dt.days.to_numpy()
        gaps = []
        if len(offsets):
            gaps.append(max(int(offsets[0]), 0))
            if len(offsets) > 1:
                gaps.extend(np.maximum(np.diff(offsets) - 1, 0).tolist())
            gaps.append(max((cutoff.normalize() - dates.iloc[-1]).days, 0))

        inter_h = g["started_at"].diff().dt.total_seconds() / 3600

        rows.append({
            "user_id": uid,
            "media_sessions_d14": len(g),
            "media_active_days_d14": g["date"].nunique(),
            "media_unique_resources_d14": g["resource_id"].nunique(),
            "media_avg_watch_pct_d14": g["watch_pct"].mean(),
            "media_median_watch_pct_d14": g["watch_pct"].median(),
            "media_watch80_rate_d14": g["watched80"].mean(),
            "media_watch50_rate_d14": g["watched50"].mean(),
            "media_live_sessions_d14": g["is_live"].sum(),
            "media_live_rate_d14": g["is_live"].mean(),
            "media_first_delay_days_d14": (
                g["started_at"].iloc[0] - start
            ).total_seconds() / 86400,
            "media_recency_days_d14": (
                cutoff - g["started_at"].iloc[-1]
            ).total_seconds() / 86400,
            "media_max_inactivity_gap_d14": max(gaps) if gaps else observation_days,
            "media_sessions_per_active_day_d14": len(g) / g["date"].nunique(),
            "media_weekend_share_d14": (g["started_at"].dt.dayofweek >= 5).mean(),
            "media_evening_share_d14": g["started_at"].dt.hour.between(18, 23).mean(),
            "media_median_inter_session_h_d14": (
                inter_h.median() if inter_h.notna().any() else 0
            ),
            "media_sessions_w1": len(w1),
            "media_sessions_w2": len(w2),
            "media_active_days_w1": w1["date"].nunique(),
            "media_active_days_w2": w2["date"].nunique(),
            "media_unique_resources_w1": w1["resource_id"].nunique(),
            "media_unique_resources_w2": w2["resource_id"].nunique(),
            "media_avg_watch_w1": w1["watch_pct"].mean() if len(w1) else 0,
            "media_avg_watch_w2": w2["watch_pct"].mean() if len(w2) else 0,
            "media_sessions_last3d": len(last3),
            "media_active_last3d": int(len(last3) > 0),
            "media_session_trend_w2_minus_w1": len(w2) - len(w1),
            "media_active_day_trend_w2_minus_w1": (
                w2["date"].nunique() - w1["date"].nunique()
            ),
        })

    f = pd.DataFrame(rows)
    out = base.merge(f, on="user_id", how="left")

    feature_cols = [c for c in f.columns if c != "user_id"]
    out[feature_cols] = out[feature_cols].fillna(0)

    if "media_sessions_d14" in out:
        zero = out["media_sessions_d14"].eq(0)
        for col in [
            "media_first_delay_days_d14",
            "media_recency_days_d14",
            "media_max_inactivity_gap_d14",
        ]:
            out.loc[zero, col] = observation_days

    return out
