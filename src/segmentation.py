"""
segmentation.py
----------------
RFM-подобная сегментация студентов вместо бинарного "успешные vs
отчисленные". Классический RFM (Recency / Frequency / Monetary) адаптирован
под образовательный продукт, где нет денежной метрики:

    R (Recency)   — сколько дней прошло с последней активности
                    (просмотр видео, ответ на задачу, тренинг) до даты среза.
    F (Frequency) — сколько дней студент был активен хоть в чем-то
                    (частота вовлечения, а не разовые всплески).
    E (Engagement depth, замена Monetary) — насколько глубоко студент
                    вовлечен: доля решенных обязательных задач, % досмотра
                    видео, средний балл за тренинги.

Сегменты: Champions, Engaged, At Risk, Dormant, New users.
"""

import numpy as np
import pandas as pd

from src.utils import clean_id, watch_pct


def compute_last_activity(data: dict) -> pd.DataFrame:
    """
    Собирает по каждому user_id дату последней активности и число уникальных
    активных дней, используя все источники активности, где есть таймстемп:
    user_activity_histories.created_at, wk_media_view_sessions.started_at,
    user_access_histories (если есть поле с таймстемпом).
    """
    frames = []

    if "user_activity_histories" in data and "user_lessons" in data:
        activity = data["user_activity_histories"].copy()
        activity["created_at"] = pd.to_datetime(activity["created_at"], errors="coerce")
        user_lessons = data["user_lessons"][["id", "user_id"]].copy()
        user_lessons["user_id"] = clean_id(user_lessons["user_id"])
        activity = activity.merge(
            user_lessons, left_on="user_lesson_id", right_on="id", how="left"
        )
        frames.append(activity[["user_id", "created_at"]].rename(columns={"created_at": "ts"}))

    if "wk_media_view_sessions" in data:
        views = data["wk_media_view_sessions"].copy()
        views["user_id"] = clean_id(views["viewer_id"])
        views["started_at"] = pd.to_datetime(views["started_at"], errors="coerce")
        frames.append(views[["user_id", "started_at"]].rename(columns={"started_at": "ts"}))

    if not frames:
        raise ValueError("Не найдено ни одного источника активности с таймстемпом.")

    events = pd.concat(frames, ignore_index=True).dropna(subset=["user_id", "ts"])
    agg = events.groupby("user_id").agg(
        last_activity=("ts", "max"),
        first_activity=("ts", "min"),
        active_days=("ts", lambda s: s.dt.date.nunique()),
    )
    return agg.reset_index()


def compute_engagement_depth(data: dict) -> pd.DataFrame:
    """
    Композитный показатель "глубины" вовлечения на пользователя:
    доля решенных задач, средний % досмотра видео, средний балл тренингов.
    Каждая компонента нормируется в [0, 1] и усредняется.
    """
    parts = []

    if "user_answers" in data:
        ans = data["user_answers"].copy()
        ans["user_id"] = clean_id(ans["user_id"])
        ans["solved_num"] = ans["solved"].map({True: 1, False: 0}).fillna(0)
        parts.append(ans.groupby("user_id")["solved_num"].mean().rename("solved_rate"))

    if "wk_media_view_sessions" in data:
        views = data["wk_media_view_sessions"].copy()
        views["user_id"] = clean_id(views["viewer_id"])
        views["watch_pct"] = watch_pct(views["viewed_segments_count"], views["segments_total"])
        parts.append(views.groupby("user_id")["watch_pct"].mean().div(100).rename("avg_watch_pct"))

    if "user_trainings" in data:
        tr = data["user_trainings"].copy()
        tr["user_id"] = clean_id(tr["user_id"])
        parts.append(tr.groupby("user_id")["mark"].mean().div(5).rename("avg_training_mark"))

    df = pd.concat(parts, axis=1)
    df["engagement_depth"] = df.mean(axis=1, skipna=True)
    return df.reset_index()


def build_rfm_table(data: dict, as_of_date, new_user_tenure_days: int = 14) -> pd.DataFrame:
    """
    Собирает итоговую RFM-таблицу на дату среза `as_of_date`.

    Важно: `as_of_date` должна совпадать (или быть раньше) с датой среза,
    используемой для риск-скоринга (см. src/features_no_leak.py), иначе
    сегментация будет "знать будущее" относительно модели.
    """
    as_of_date = pd.to_datetime(as_of_date)

    activity = compute_last_activity(data)
    engagement = compute_engagement_depth(data)

    rfm = activity.merge(engagement, on="user_id", how="outer")
    rfm["recency_days"] = (as_of_date - rfm["last_activity"]).dt.days
    rfm["tenure_days"] = (as_of_date - rfm["first_activity"]).dt.days
    rfm["active_days"] = rfm["active_days"].fillna(0)
    rfm["engagement_depth"] = rfm["engagement_depth"].fillna(0)

    return rfm


def _tertile_score(series: pd.Series, ascending: bool) -> pd.Series:
    """1..3, где 3 = "лучше" (недавняя активность / высокая частота)."""
    ranks = series.rank(method="first")
    labels = [1, 2, 3] if ascending else [3, 2, 1]
    return pd.qcut(ranks, 3, labels=labels).astype(int)


def assign_segments(rfm: pd.DataFrame, new_user_tenure_days: int = 14) -> pd.DataFrame:
    """
    Присваивает сегмент каждому пользователю:

    New users  — tenure_days <= new_user_tenure_days (недостаточно истории
                 для честной оценки, сегментировать рано).
    Champions  — недавняя активность (R=3) и высокая частота/глубина (F=3).
    Engaged    — стабильно вовлечены, но не топ (R>=2 и F>=2, не Champions).
    At Risk    — раньше были вовлечены (F>=2), но давно не заходили (R=1).
    Dormant    — низкая частота/глубина независимо от давности (F=1).
    """
    df = rfm.copy()

    # F-компонента: частота * глубина вовлечения, оба фактора имеют значение
    df["freq_engagement_raw"] = df["active_days"].rank(pct=True) * 0.5 + \
                                 df["engagement_depth"].rank(pct=True) * 0.5

    df["R_score"] = _tertile_score(df["recency_days"], ascending=False)  # меньше recency_days = недавняя активность = выше score
    df["F_score"] = _tertile_score(df["freq_engagement_raw"], ascending=True)

    def label(row):
        if row["tenure_days"] <= new_user_tenure_days:
            return "New users"
        if row["R_score"] == 3 and row["F_score"] == 3:
            return "Champions"
        if row["R_score"] >= 2 and row["F_score"] >= 2:
            return "Engaged"
        if row["R_score"] == 1 and row["F_score"] >= 2:
            return "At Risk"
        return "Dormant"

    df["segment"] = df.apply(label, axis=1)
    return df
