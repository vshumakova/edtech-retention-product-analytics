"""
funnel.py
---------
Строит продуктовую воронку обучения в двух разрезах:

1. "Внутри модуля" (within-module funnel) — от старта модуля до его
   завершения, по шагам вовлечения: посмотрел урок -> посетил онлайн-занятие
   -> решил обязательные задачи -> прошел текущий контроль -> сдал ПА ->
   завершил модуль.

2. "Между модулями" (cross-module funnel) — сквозная воронка программы:
   зарегистрировался -> записался на курс -> начал Модуль 1 -> завершил
   Модуль 1 -> начал Модуль 2 -> ... -> завершил Модуль 4.

Business logic: раньше проект сравнивал только "успешные vs отчисленные"
после факта. Воронка показывает, НА КАКОМ шаге отваливается основная масса
пользователей — это то, что продуктовая команда может чинить точечно.
"""

from dataclasses import dataclass
import pandas as pd

from src.utils import clean_id, yes_no_to_flag


@dataclass
class FunnelStep:
    name: str
    mask: "pd.Series"  # булева маска относительно исходного df


def within_module_funnel(df_module: pd.DataFrame) -> pd.DataFrame:
    """
    Строит воронку из одного stats__module_N датафрейма.

    Шаги подобраны по колонкам, которые реально есть в stats__module_*:
    'Просмотрел уроков', 'Посетил урок в онлайне', 'Решены все обяз.ИЗ',
    'Пройден тек.контроль', 'Сдал ПА', 'Статус'.
    """
    df = df_module.copy()
    df["user_id"] = clean_id(df["user_id"])

    n_started = len(df)

    steps = [
        FunnelStep("Начал модуль", pd.Series(True, index=df.index)),
        FunnelStep("Посмотрел ≥1 урок", df["Просмотрел уроков"].fillna(0) > 0),
        FunnelStep("Посетил онлайн-занятие", yes_no_to_flag(df["Посетил урок в онлайне"]) == 1),
        FunnelStep("Решил все обязательные ИЗ", yes_no_to_flag(df["Решены все обяз.ИЗ"]) == 1),
        FunnelStep("Прошел текущий контроль", df["Пройден тек.контроль"] == "Да"),
        FunnelStep("Сдал промежуточную аттестацию", df["Сдал ПА"] == "Да"),
        FunnelStep("Завершил модуль", df["Статус"] == "Завершил"),
    ]

    rows = []
    prev_count = None
    for step in steps:
        count = int(step.mask.sum())
        conv_from_start = count / n_started * 100 if n_started else 0
        conv_from_prev = (count / prev_count * 100) if prev_count else 100.0
        drop_off = prev_count - count if prev_count is not None else 0
        rows.append({
            "step": step.name,
            "users": count,
            "conversion_from_start_%": round(conv_from_start, 1),
            "conversion_from_prev_step_%": round(conv_from_prev, 1),
            "drop_off_users": drop_off,
        })
        prev_count = count

    return pd.DataFrame(rows)


def cross_module_funnel(data: dict, module_status_tables: dict) -> pd.DataFrame:
    """
    Сквозная воронка программы: сколько пользователей дошли до старта
    и до завершения каждого модуля.

    module_status_tables: результат data_loading.get_module_status_tables(data)
    """
    rows = []

    n_registered = data["users"]["id"].nunique() if "id" in data["users"].columns else len(data["users"])
    rows.append({"step": "Зарегистрировался", "users": n_registered})

    if "users_courses" in data:
        n_enrolled = clean_id(data["users_courses"]["user_id"]).nunique()
        rows.append({"step": "Записался на курс", "users": n_enrolled})

    for name, df in module_status_tables.items():
        module_num = df["module"].iloc[0]
        uids = clean_id(df["user_id"]).dropna()
        started = uids.nunique()
        completed = clean_id(df.loc[df["Статус"] == "Завершил", "user_id"]).nunique()
        rows.append({"step": f"Начал Модуль {module_num}", "users": started})
        rows.append({"step": f"Завершил Модуль {module_num}", "users": completed})

    funnel = pd.DataFrame(rows)
    base = funnel["users"].iloc[0]
    funnel["conversion_from_start_%"] = (funnel["users"] / base * 100).round(1)
    funnel["conversion_from_prev_step_%"] = (
        funnel["users"] / funnel["users"].shift(1) * 100
    ).fillna(100).round(1)
    return funnel


def north_star_metric(module_status_tables: dict) -> pd.DataFrame:
    """
    North Star Metric: Course Completion Rate (CCR) по каждому модулю —
    доля студентов, завершивших модуль, среди начавших его.

    Считается по модулям отдельно (входные когорты в module N — это уже
    отфильтрованные "выжившие" module N-1, поэтому не усредняем в одну
    цифру напрямую, а показываем тренд по шагам программы).
    """
    rows = []
    for name, df in module_status_tables.items():
        module_num = df["module"].iloc[0]
        started = len(df)
        completed = (df["Статус"] == "Завершил").sum()
        ccr = completed / started * 100 if started else 0
        rows.append({"module": module_num, "started": started, "completed": completed, "CCR_%": round(ccr, 1)})
    return pd.DataFrame(rows)
