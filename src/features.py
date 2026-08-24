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
    g = groups.copy()
    g["parallel_id"] = pd.to_numeric(normalize_id(g["group_template_id"]), errors="coerce")
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
    s = stats_m1[stats_m1["user_id"].notna()].copy()
    s["user_id"] = normalize_id(s["user_id"])
    s["parallel_id"] = pd.to_numeric(s["id параллели"], errors="coerce")
    s["enrollment_date"] = pd.to_datetime(s["Дата зачисления"], errors="coerce")
    s["churn"] = s["Статус"].map({"Отчислен": 1, "Завершил": 0})

    s = (
        s.sort_values("user_id")
        .drop_duplicates("user_id", keep="first")
        [["user_id", "parallel_id", "enrollment_date", "churn", "Уровень", "course_id"]]
    )

    s = s.merge(build_parallel_start(groups), on="parallel_id", how="left")
    s = s.dropna(subset=["learning_start", "churn"]).copy()
    s["churn"] = s["churn"].astype(int)
    s["cutoff_d14"] = s["learning_start"] + pd.Timedelta(days=observation_days)

    s["enrollment_lead_days"] = (
        s["learning_start"] - s["enrollment_date"]
    ).dt.total_seconds() / 86400
    s["start_weekday"] = s["learning_start"].dt.dayofweek.astype("string")
    s["start_hour"] = s["learning_start"].dt.hour

    cohort_sizes = []
    for pid, g in s.groupby("parallel_id"):
        start = g["learning_start"].iloc[0]
        cohort_sizes.append((pid, int((g["enrollment_date"] <= start).sum())))

    s = s.merge(
        pd.DataFrame(cohort_sizes, columns=["parallel_id", "cohort_size_at_start"]),
        on="parallel_id",
        how="left",
    )
    return s


def build_d14_media_features(
    base: pd.DataFrame,
    media: pd.DataFrame,
    observation_days: int = 14,
) -> pd.DataFrame:
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
        raise AssertionError("Future media event detected.")

    d["date"] = d["started_at"].dt.normalize()
    denom = pd.to_numeric(d["segments_total"], errors="coerce").replace(0, np.nan)
    viewed = pd.to_numeric(d["viewed_segments_count"], errors="coerce")
    d["watch_pct"] = (viewed / denom).clip(0, 1)
    d["watched80"] = (d["watch_pct"] >= 0.80).astype(int)
    d["watched50"] = (d["watch_pct"] >= 0.50).astype(int)
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
            "media_first_delay_days_d14": (g["started_at"].iloc[0] - start).total_seconds() / 86400,
            "media_recency_days_d14": (cutoff - g["started_at"].iloc[-1]).total_seconds() / 86400,
            "media_max_inactivity_gap_d14": max(gaps) if gaps else observation_days,
            "media_sessions_per_active_day_d14": len(g) / g["date"].nunique(),
            "media_weekend_share_d14": (g["started_at"].dt.dayofweek >= 5).mean(),
            "media_evening_share_d14": g["started_at"].dt.hour.between(18, 23).mean(),
            "media_median_inter_session_h_d14": inter_h.median() if inter_h.notna().any() else 0,
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

    zero = out["media_sessions_d14"].eq(0)
    for col in [
        "media_first_delay_days_d14",
        "media_recency_days_d14",
        "media_max_inactivity_gap_d14",
    ]:
        out.loc[zero, col] = observation_days
    return out


def build_d14_training_features(
    base: pd.DataFrame,
    user_trainings: pd.DataFrame,
    trainings: pd.DataFrame,
    observation_days: int = 14,
) -> pd.DataFrame:
    """
    Training-start events are safe when started_at <= D14.

    Outcome-like fields such as solved_tasks_count, earned_points and mark
    are used only when their corresponding completion/mark timestamp is <= D14.
    """
    u = user_trainings.copy()
    u["user_id"] = normalize_id(u["user_id"])
    u["training_id_clean"] = pd.to_numeric(normalize_id(u["training_id"]), errors="coerce")

    for c in ["started_at", "finished_at", "mark_saved_at"]:
        u[c] = pd.to_datetime(u[c], errors="coerce")

    meta = trainings.copy()
    meta["training_id_clean"] = pd.to_numeric(normalize_id(meta["id"]), errors="coerce")
    meta = meta[["training_id_clean", "difficulty", "task_templates_count"]]

    u = u.merge(meta, on="training_id_clean", how="left")

    d = u.merge(
        base[["user_id", "learning_start", "cutoff_d14"]],
        on="user_id",
        how="inner",
        validate="many_to_one",
    )

    d = d[
        d["started_at"].notna()
        & (d["started_at"] >= d["learning_start"])
        & (d["started_at"] <= d["cutoff_d14"])
    ].copy()

    d["finished_by_d14"] = (
        d["finished_at"].notna() & (d["finished_at"] <= d["cutoff_d14"])
    )
    d["marked_by_d14"] = (
        d["mark_saved_at"].notna() & (d["mark_saved_at"] <= d["cutoff_d14"])
    )
    d["date"] = d["started_at"].dt.normalize()
    d["day_idx"] = (d["started_at"] - d["learning_start"]).dt.total_seconds() / 86400

    rows = []
    for uid, g in d.groupby("user_id", sort=False):
        start = g["learning_start"].iloc[0]
        cutoff = g["cutoff_d14"].iloc[0]

        completed = g[g["finished_by_d14"]]
        marked = g[g["marked_by_d14"]]
        w1 = g[g["day_idx"] < 7]
        w2 = g[g["day_idx"] >= 7]
        last3 = g[g["day_idx"] >= 11]
        types = g["type"].astype("string").value_counts()

        rows.append({
            "user_id": uid,
            "tr_started_d14": len(g),
            "tr_unique_trainings_d14": g["training_id_clean"].nunique(),
            "tr_active_days_d14": g["date"].nunique(),
            "tr_completed_d14": len(completed),
            "tr_completion_rate_d14": len(completed) / len(g),
            "tr_solved_tasks_d14": pd.to_numeric(
                completed["solved_tasks_count"], errors="coerce"
            ).fillna(0).sum(),
            "tr_submitted_answers_d14": pd.to_numeric(
                completed["submitted_answers_count"], errors="coerce"
            ).fillna(0).sum(),
            "tr_earned_points_d14": pd.to_numeric(
                completed["earned_points"], errors="coerce"
            ).fillna(0).sum(),
            "tr_avg_attempts_completed_d14": pd.to_numeric(
                completed["attempts"], errors="coerce"
            ).mean() if len(completed) else 0,
            "tr_marked_count_d14": len(marked),
            "tr_avg_mark_d14": pd.to_numeric(
                marked["mark"], errors="coerce"
            ).mean() if len(marked) else 0,
            "tr_first_delay_days_d14": (g["started_at"].min() - start).total_seconds() / 86400,
            "tr_recency_days_d14": (cutoff - g["started_at"].max()).total_seconds() / 86400,
            "tr_started_w1": len(w1),
            "tr_started_w2": len(w2),
            "tr_started_last3d": len(last3),
            "tr_week2_share": len(w2) / len(g),
            "tr_lesson_training_count_d14": int(
                types.get("UserTrainings::LessonTraining", 0)
            ),
            "tr_regular_training_count_d14": int(
                types.get("UserTrainings::RegularTraining", 0)
            ),
            "tr_olympiad_training_count_d14": int(
                types.get("UserTrainings::OlympiadTraining", 0)
            ),
            "tr_avg_difficulty_d14": pd.to_numeric(
                g["difficulty"], errors="coerce"
            ).mean(),
            "tr_task_capacity_d14": pd.to_numeric(
                g["task_templates_count"], errors="coerce"
            ).fillna(0).sum(),
        })

    f = pd.DataFrame(rows)
    out = base[["user_id"]].merge(f, on="user_id", how="left")
    feature_cols = [c for c in f.columns if c != "user_id"]
    out[feature_cols] = out[feature_cols].fillna(0)

    zero = out["tr_started_d14"].eq(0)
    out.loc[zero, "tr_first_delay_days_d14"] = observation_days
    out.loc[zero, "tr_recency_days_d14"] = observation_days
    out["tr_has_activity_d14"] = (~zero).astype(int)
    return out


def build_selected_feature_store(
    stats_m1: pd.DataFrame,
    groups: pd.DataFrame,
    media: pd.DataFrame,
    user_trainings: pd.DataFrame,
    trainings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final operating feature store.

    Source ablation showed that media + training produces the strongest
    Top-20% targeting metrics. Course/access tables are audited separately
    but intentionally excluded from the selected operating model.
    """
    base = build_student_base(stats_m1, groups)
    media_df = build_d14_media_features(base, media)
    training_df = build_d14_training_features(base, user_trainings, trainings)

    training_cols = [c for c in training_df.columns if c != "user_id"]
    return media_df.merge(
        training_df[["user_id"] + training_cols],
        on="user_id",
        how="left",
    )
