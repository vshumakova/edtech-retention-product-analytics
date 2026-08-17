"""
features_no_leak.py
--------------------
КРИТИЧЕСКИ ВАЖНЫЙ модуль: здесь исправлена главная методологическая ошибка
исходного ноутбука — data leakage.

Что было не так:
    Модель обучалась на признаках вроде "Решено ИЗ", "Решены все обяз.ИЗ",
    "Пройден тек.контроль", "tr_total_solved" — это АГРЕГАТЫ ЗА ВЕСЬ МОДУЛЬ,
    то есть буквально то же самое событие, что и целевая переменная
    (Статус = Завершил/Отчислен), только в числовом виде. Отсюда ROC-AUC
    0.998 — модель не предсказывала риск, а пересказывала факт.
    Пример: "Решены все обяз.ИЗ" = 'Нет' => 100% Отчислен по построению.

Что делаем вместо этого:
    Фиксируем точку отсечки (cutoff) относительно даты зачисления студента,
    например "первые 14 дней обучения", и считаем признаки ТОЛЬКО по
    событиям с таймстемпом раньше этой точки. Целевая переменная
    (завершил/отчислен) по определению становится известна позже cutoff —
    так модель учится именно на раннем сигнале, который можно использовать
    для риск-скоринга в реальном времени, а не на пересказе итога.

    Признаки из stats__module_* НЕ используются вообще (кроме даты
    зачисления и уровня курса, которые известны в момент t=0) — они
    являются финальными агрегатами и не привязаны к моменту времени.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from src.utils import clean_id, watch_pct, status_to_target


def _enrollment_dates(module_status_tables: dict) -> pd.DataFrame:
    """user_id -> дата зачисления, уровень курса, модуль, target (известен только постфактум)."""
    frames = []
    for name, df in module_status_tables.items():
        tmp = pd.DataFrame({
            "user_id": clean_id(df["user_id"]),
            "module": df["module"],
            "enrolled_at": pd.to_datetime(df.get("Дата зачисления"), errors="coerce"),
            "level": df.get("Уровень"),
            "target": status_to_target(df["Статус"]),
        })
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True).dropna(subset=["user_id", "enrolled_at"])


def _windowed_activity_features(data: dict, base: pd.DataFrame, cutoff_days: int) -> pd.DataFrame:
    """
    Признаки из user_activity_histories, посчитанные только по событиям
    в окне [enrolled_at, enrolled_at + cutoff_days].
    """
    if "user_activity_histories" not in data or "user_lessons" not in data:
        return base[["user_id", "module"]].copy()

    activity = data["user_activity_histories"].copy()
    activity["created_at"] = pd.to_datetime(activity["created_at"], errors="coerce")
    user_lessons = data["user_lessons"][["id", "user_id"]].copy()
    user_lessons["user_id"] = clean_id(user_lessons["user_id"])
    activity = activity.merge(user_lessons, left_on="user_lesson_id", right_on="id", how="left")

    merged = activity.merge(base[["user_id", "module", "enrolled_at"]], on="user_id", how="inner")
    cutoff = merged["enrolled_at"] + pd.to_timedelta(cutoff_days, unit="D")
    windowed = merged[(merged["created_at"] >= merged["enrolled_at"]) & (merged["created_at"] <= cutoff)]

    agg = windowed.groupby(["user_id", "module"]).agg(
        early_actions_total=("action", "count"),
        early_active_days=("created_at", lambda s: s.dt.date.nunique()),
    )
    action_dummies = pd.get_dummies(windowed["action"], prefix="early_action")
    action_counts = pd.concat(
        [windowed[["user_id", "module"]], action_dummies], axis=1
    ).groupby(["user_id", "module"]).sum()

    out = agg.join(action_counts, how="outer").reset_index()
    return out


def _windowed_media_features(data: dict, base: pd.DataFrame, cutoff_days: int) -> pd.DataFrame:
    """Признаки из wk_media_view_sessions в раннем окне."""
    if "wk_media_view_sessions" not in data:
        return base[["user_id", "module"]].copy()

    views = data["wk_media_view_sessions"].copy()
    views["user_id"] = clean_id(views["viewer_id"])
    views["started_at"] = pd.to_datetime(views["started_at"], errors="coerce")
    views["watch_pct"] = watch_pct(views["viewed_segments_count"], views["segments_total"])

    merged = views.merge(base[["user_id", "module", "enrolled_at"]], on="user_id", how="inner")
    cutoff = merged["enrolled_at"] + pd.to_timedelta(cutoff_days, unit="D")
    windowed = merged[(merged["started_at"] >= merged["enrolled_at"]) & (merged["started_at"] <= cutoff)]

    is_live = (windowed["kind"] == "ulms_live").astype(int) if "kind" in windowed.columns else 0

    out = windowed.assign(is_live=is_live).groupby(["user_id", "module"]).agg(
        early_views_total=("watch_pct", "count"),
        early_avg_watch_pct=("watch_pct", "mean"),
        early_live_views=("is_live", "sum"),
    ).reset_index()
    return out


def build_leakage_safe_features(
    data: dict,
    module_status_tables: dict,
    cutoff_days: int = 14,
) -> tuple[pd.DataFrame, list]:
    """
    Собирает признаковую матрицу, где КАЖДЫЙ признак вычислен только по
    данным до `enrolled_at + cutoff_days`. Возвращает (df, feature_columns).

    Целевая переменная (`target`) остается финальным статусом модуля —
    она наблюдается позже cutoff, поэтому используется только как label,
    никогда как признак.

    Note: user_answers и user_trainings в этом датасете не содержат
    надежной колонки с таймстемпом на уровне события, поэтому в этой
    версии они не используются в риск-скоринге, чтобы не гадать про
    момент события. Если в вашей выгрузке такая колонка есть (например
    created_at в user_trainings) — добавьте аналогичную windowed-функцию
    по образцу _windowed_media_features.
    """
    base = _enrollment_dates(module_status_tables)

    activity_feat = _windowed_activity_features(data, base, cutoff_days)
    media_feat = _windowed_media_features(data, base, cutoff_days)

    df = base.merge(activity_feat, on=["user_id", "module"], how="left")
    df = df.merge(media_feat, on=["user_id", "module"], how="left")

    feature_cols = [c for c in df.columns if c.startswith("early_")]
    df[feature_cols] = df[feature_cols].fillna(0)

    # Уровень курса — известен на входе, это не утечка.
    if "level" in df.columns:
        level_dummies = pd.get_dummies(df["level"], prefix="level")
        df = pd.concat([df, level_dummies], axis=1)
        feature_cols += list(level_dummies.columns)

    df["module"] = df["module"].astype(int)
    feature_cols.append("module")

    return df, feature_cols
