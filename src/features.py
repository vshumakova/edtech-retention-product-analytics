from __future__ import annotations

import re
import numpy as np
import pandas as pd


FORBIDDEN_D14_COLUMNS = {
    "Решено ИЗ",
    "Решены все обяз.ИЗ",
    "Пройден тек.контроль",
    "Балл ПА",
    "Сдал ПА",
    "Дата сдачи ПА (МСК)",
    "Статус",
    "tr_total_solved",
    "solved_tasks_count",
    "earned_points",
    "mark",
}


def normalize_id(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def assert_no_forbidden_features(columns) -> None:
    bad = sorted(set(columns) & FORBIDDEN_D14_COLUMNS)
    if bad:
        raise ValueError(f"Non-D14 / outcome columns detected: {bad}")


def choose_time_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing event-time column from an explicit allowlist."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def window_events(
    events: pd.DataFrame,
    enrollment: pd.DataFrame,
    event_user_col: str,
    event_time_col: str,
    enrollment_user_col: str = "user_id",
    enrollment_time_col: str = "Дата зачисления",
    days: int = 14,
) -> pd.DataFrame:
    """Point-in-time filter. No event after the D14 cutoff can survive."""
    e = events.copy()
    enr = enrollment[[enrollment_user_col, enrollment_time_col]].drop_duplicates().copy()

    e[event_user_col] = normalize_id(e[event_user_col])
    enr[enrollment_user_col] = normalize_id(enr[enrollment_user_col])
    e[event_time_col] = pd.to_datetime(e[event_time_col], errors="coerce")
    enr[enrollment_time_col] = pd.to_datetime(enr[enrollment_time_col], errors="coerce")

    e = e.dropna(subset=[event_time_col])
    out = e.merge(
        enr,
        left_on=event_user_col,
        right_on=enrollment_user_col,
        how="inner",
        validate="many_to_one",
    )
    out["cutoff_d14"] = out[enrollment_time_col] + pd.Timedelta(days=days)
    out = out[
        (out[event_time_col] >= out[enrollment_time_col])
        & (out[event_time_col] <= out["cutoff_d14"])
    ].copy()

    if len(out) and not (out[event_time_col] <= out["cutoff_d14"]).all():
        raise AssertionError("Point-in-time violation: future event in D14 feature set.")
    return out


def _safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(x).lower()).strip("_")


def aggregate_activity_d14(
    activity: pd.DataFrame,
    user_lessons: pd.DataFrame,
    enrollment: pd.DataFrame,
    days: int = 14,
) -> pd.DataFrame:
    """Behavioral cadence features from timestamped LMS activity."""
    act = activity.copy()
    ul = user_lessons[["id", "user_id", "lesson_id"]].copy()

    act["user_lesson_id"] = normalize_id(act["user_lesson_id"])
    ul["id"] = normalize_id(ul["id"])
    ul["user_id"] = normalize_id(ul["user_id"])

    act = act.merge(
        ul,
        left_on="user_lesson_id",
        right_on="id",
        how="inner",
        validate="many_to_one",
    )

    d = window_events(
        act,
        enrollment,
        event_user_col="user_id",
        event_time_col="created_at",
        days=days,
    )
    if d.empty:
        return pd.DataFrame(columns=["user_id"])

    d["date"] = d["created_at"].dt.normalize()
    rows = []

    for uid, g in d.groupby("user_id"):
        g = g.sort_values("created_at")
        t0 = g["Дата зачисления"].iloc[0]
        cutoff = g["cutoff_d14"].iloc[0]
        times = g["created_at"]
        active_dates = g["date"].drop_duplicates().sort_values()

        session_break = times.diff().dt.total_seconds().div(60) > 30
        sessions = 1 + int(session_break.sum())

        offsets = (active_dates - t0.normalize()).dt.days.to_numpy()
        gaps = []
        if len(offsets):
            gaps.append(max(int(offsets[0]), 0))
            if len(offsets) > 1:
                gaps.extend(np.maximum(np.diff(offsets) - 1, 0).tolist())
            gaps.append(max((cutoff.normalize() - active_dates.iloc[-1]).days, 0))

        row = {
            "user_id": uid,
            "act_events_d14": len(g),
            "act_active_days_d14": g["date"].nunique(),
            "act_inactive_days_d14": days - g["date"].nunique(),
            "act_sessions_d14": sessions,
            "act_unique_lessons_d14": g["lesson_id"].nunique(),
            "act_first_delay_h_d14": (times.iloc[0] - t0).total_seconds() / 3600,
            "act_recency_days_d14": (cutoff - times.iloc[-1]).total_seconds() / 86400,
            "act_max_inactivity_gap_d14": max(gaps) if gaps else days,
            "act_events_per_active_day_d14": len(g) / max(g["date"].nunique(), 1),
            "act_weekend_share_d14": float((times.dt.dayofweek >= 5).mean()),
            "act_evening_share_d14": float(times.dt.hour.between(18, 23).mean()),
        }

        for action, count in g["action"].astype("string").value_counts().items():
            n = _safe_name(action)
            row[f"act_{n}_count_d14"] = int(count)
            row[f"act_{n}_share_d14"] = float(count / len(g))
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_answers_d14(
    answers: pd.DataFrame,
    enrollment: pd.DataFrame,
    time_candidates=("created_at", "answered_at", "updated_at"),
    days: int = 14,
) -> pd.DataFrame:
    """Task behavior from answer events, only when an event timestamp exists."""
    time_col = choose_time_column(answers, list(time_candidates))
    if time_col is None:
        return pd.DataFrame(columns=["user_id"])

    a = answers.copy()
    a["user_id"] = normalize_id(a["user_id"])
    d = window_events(a, enrollment, "user_id", time_col, days=days)
    if d.empty:
        return pd.DataFrame(columns=["user_id"])

    if "solved" in d:
        d["solved_num"] = d["solved"].map({True: 1, False: 0}).fillna(
            pd.to_numeric(d["solved"], errors="coerce")
        ).fillna(0)
    else:
        d["solved_num"] = 0

    if "skipped" in d:
        d["skipped_num"] = d["skipped"].map({True: 1, False: 0}).fillna(
            pd.to_numeric(d["skipped"], errors="coerce")
        ).fillna(0)
    else:
        d["skipped_num"] = 0

    agg_spec = {
        "task_id": ["count", "nunique"],
        "solved_num": ["mean", "sum"],
        "skipped_num": ["mean"],
    }
    if "attempts" in d:
        agg_spec["attempts"] = ["mean", "max"]

    out = d.groupby("user_id").agg(agg_spec)
    out.columns = [
        "ans_events_d14",
        "ans_unique_tasks_d14",
        "ans_solved_rate_d14",
        "ans_solved_count_d14",
        "ans_skipped_rate_d14",
    ] + (["ans_avg_attempts_d14", "ans_max_attempts_d14"] if "attempts" in d else [])
    return out.reset_index()


def aggregate_media_d14(
    media: pd.DataFrame,
    enrollment: pd.DataFrame,
    time_candidates=("created_at", "started_at", "viewed_at", "updated_at"),
    days: int = 14,
) -> pd.DataFrame:
    """Media engagement, used only if a session timestamp exists."""
    time_col = choose_time_column(media, list(time_candidates))
    if time_col is None or "viewer_id" not in media:
        return pd.DataFrame(columns=["user_id"])

    m = media.copy()
    m["viewer_id"] = normalize_id(m["viewer_id"])
    d = window_events(m, enrollment, "viewer_id", time_col, days=days)
    if d.empty:
        return pd.DataFrame(columns=["user_id"])

    denom = pd.to_numeric(d["segments_total"], errors="coerce").replace(0, np.nan)
    d["watch_pct"] = (
        pd.to_numeric(d["viewed_segments_count"], errors="coerce") / denom
    ).clip(0, 1)
    d["watched_80"] = (d["watch_pct"] >= 0.80).astype(int)

    named = d.rename(columns={"viewer_id": "user_id"})
    out = named.groupby("user_id").agg(
        media_sessions_d14=("watch_pct", "size"),
        media_avg_watch_pct_d14=("watch_pct", "mean"),
        media_watch_80_rate_d14=("watched_80", "mean"),
        media_resources_d14=("resource_id", "nunique"),
    ).reset_index()

    if "kind" in named:
        live = (
            named.assign(is_live=named["kind"].eq("ulms_live").astype(int))
            .groupby("user_id", as_index=False)["is_live"].sum()
            .rename(columns={"is_live": "media_live_sessions_d14"})
        )
        out = out.merge(live, on="user_id", how="left")
    return out


def build_feature_matrix(
    stats_m1: pd.DataFrame,
    activity: pd.DataFrame,
    user_lessons: pd.DataFrame,
    answers: pd.DataFrame | None = None,
    media: pd.DataFrame | None = None,
    days: int = 14,
) -> pd.DataFrame:
    """Combine safe D14 features and preserve zero-activity students."""
    enrollment_cols = ["user_id", "Дата зачисления"]
    for c in ["Уровень", "teacher_id", "id параллели"]:
        if c in stats_m1:
            enrollment_cols.append(c)

    enrollment = stats_m1[enrollment_cols].drop_duplicates("user_id").copy()
    enrollment["user_id"] = normalize_id(enrollment["user_id"])

    base = enrollment.copy()
    pieces = [aggregate_activity_d14(activity, user_lessons, enrollment, days)]

    if answers is not None:
        pieces.append(aggregate_answers_d14(answers, enrollment, days=days))
    if media is not None:
        pieces.append(aggregate_media_d14(media, enrollment, days=days))

    for p in pieces:
        if len(p.columns) > 1:
            p = p.copy()
            p["user_id"] = normalize_id(p["user_id"])
            base = base.merge(p, on="user_id", how="left")

    numeric = base.select_dtypes(include=[np.number]).columns
    behavior_numeric = [c for c in numeric if c not in {"teacher_id", "id параллели"}]
    base[behavior_numeric] = base[behavior_numeric].fillna(0)

    # Zero activity means 14 inactive days / max gap.
    if "act_events_d14" in base:
        zero = base["act_events_d14"].eq(0)
        for c in ["act_inactive_days_d14", "act_max_inactivity_gap_d14", "act_recency_days_d14"]:
            if c in base:
                base.loc[zero, c] = days

    assert_no_forbidden_features(base.columns)
    return base
