"""
utils.py
--------
Мелкие переиспользуемые функции очистки данных. Вынесены из оригинального
ноутбука, где одна и та же логика (очистка user_id, расчет watch_pct)
копипастилась в 4-5 разных функциях с небольшими отличиями — источник
потенциальных рассинхронов при рефакторинге.
"""

import numpy as np
import pandas as pd


def clean_id(series: pd.Series) -> pd.Series:
    """
    Приводит id-колонку к float, устраняя артефакты вида "12,345".

    Исходные CSV хранят user_id то как int, то как строку с разделителем
    тысяч — из-за этого прямые merge/isin по user_id молча теряли строки.
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", "").str.strip(),
        errors="coerce",
    )


def watch_pct(viewed_segments: pd.Series, segments_total: pd.Series) -> pd.Series:
    """Процент досмотра видео/контента, ограниченный [0, 100]."""
    return (viewed_segments / segments_total.replace(0, np.nan) * 100).clip(0, 100).fillna(0)


def status_to_target(status_series: pd.Series, success_label: str = "Завершил") -> pd.Series:
    """Бинарная целевая переменная: 1 = удержан (завершил модуль), 0 = отчислен."""
    return (status_series == success_label).astype(int)


def yes_no_to_flag(series: pd.Series, yes_label: str = "Да") -> pd.Series:
    return (series == yes_label).astype(int)
