"""
cohorts.py
----------
Когортный анализ удержания.

Когорта = месяц зачисления студента (поле "Дата зачисления" в
stats__module_1 — это точка входа в программу). Дальше для каждой когорты
смотрим, какая доля дошла живой (со статусом "Завершил" на предыдущем шаге)
до старта и до завершения Модуля 1, 2, 3, 4.

Это отвечает на вопрос "как меняется удержание для более поздних когорт" —
то, чего не было в исходном ноутбуке (там между модулями не было понятия
времени вообще).
"""

import pandas as pd

from src.utils import clean_id


def build_user_module_status(module_status_tables: dict) -> pd.DataFrame:
    """
    Собирает длинную таблицу (user_id, module, status, enrolled_at) —
    по одной строке на пользователя на модуль, в который он вошел.
    """
    frames = []
    for name, df in module_status_tables.items():
        tmp = pd.DataFrame({
            "user_id": clean_id(df["user_id"]),
            "module": df["module"],
            "status": df["Статус"],
        })
        if "Дата зачисления" in df.columns:
            tmp["enrolled_at"] = pd.to_datetime(df["Дата зачисления"], errors="coerce")
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True).dropna(subset=["user_id"])


def assign_cohort(user_module_status: pd.DataFrame, cohort_freq: str = "M") -> pd.DataFrame:
    """
    Присваивает каждому пользователю когорту по дате зачисления в Модуль 1
    (самая ранняя известная дата входа в программу для этого user_id).
    """
    m1_dates = (
        user_module_status[user_module_status["module"] == 1]
        .dropna(subset=["enrolled_at"])
        .groupby("user_id")["enrolled_at"].min()
        .rename("cohort_date")
    )
    out = user_module_status.merge(m1_dates, on="user_id", how="left")
    out["cohort"] = out["cohort_date"].dt.to_period(cohort_freq).astype(str)
    return out


def cohort_module_retention(user_module_status_with_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Матрица для тепловой карты: строки — когорты (месяц зачисления),
    столбцы — модуль программы, значения — % от размера когорты,
    завершивших этот модуль (Статус == 'Завершил').

    Знаменатель — размер когорты на входе в Модуль 1, поэтому цифры
    по определению монотонно не возрастают слева направо и напрямую
    показывают "усадку" когорты по программе.
    """
    df = user_module_status_with_cohort.dropna(subset=["cohort"])
    cohort_size = (
        df[df["module"] == 1].groupby("cohort")["user_id"].nunique().rename("cohort_size")
    )

    completed = df[df["status"] == "Завершил"]
    pivot = (
        completed.groupby(["cohort", "module"])["user_id"]
        .nunique()
        .unstack("module")
        .reindex(columns=[1, 2, 3, 4])
    )

    pivot = pivot.join(cohort_size)
    retention = pivot.drop(columns="cohort_size").div(pivot["cohort_size"], axis=0) * 100
    retention.insert(0, "cohort_size", pivot["cohort_size"])
    retention.columns = ["cohort_size"] + [f"Модуль {m}" for m in [1, 2, 3, 4]]
    return retention.round(1).sort_index()
