"""
data_loading.py
----------------
Единая точка загрузки сырых данных проекта.

Дизайн-решение: раньше путь к данным был захардкожен на Google Drive
конкретного человека (`/content/drive/MyDrive/MIPT/Hackathon/Cifrium/...`).
Это ломает воспроизводимость — код нельзя запустить без доступа к чужому
диску. Здесь путь к данным вынесен в переменную окружения / параметр,
чтобы ноутбук воспроизводился и в Colab, и локально, и в CI.

Использование:
    from src.data_loading import load_all_data
    data = load_all_data(DATA_DIR)  # DATA_DIR = "data/raw" или путь к Google Drive
"""

from pathlib import Path
import pandas as pd

# Имена таблиц -> имена файлов. Список сознательно не включает таблицы,
# которые не использовались ни в одном анализе (user_award_badges,
# wk_users_courses_actions, award_badges) — если понадобятся, добавьте сюда.
TABLE_FILES = {
    "users": "users.csv",
    "users_courses": "users_courses.csv",
    "user_answers": "user_answers.csv",
    "user_activity_histories": "user_activity_histories.csv",
    "user_trainings": "user_trainings.csv",
    "user_access_histories": "user_access_histories.csv",
    "user_lessons": "user_lessons.csv",
    "wk_media_view_sessions": "wk_media_view_sessions.csv",
    "lessons": "lessons.csv",
    "lesson_tasks": "lesson_tasks.csv",
    "stats__module_1": "stats__module_1.csv",
    "stats__module_2": "stats__module_2.csv",
    "stats__module_3": "stats__module_3.csv",
    "stats__module_4": "stats__module_4.csv",
}

# Модули курса в порядке прохождения — используется во всех ноутбуках
# для построения воронки "Модуль 1 -> Модуль 2 -> Модуль 3 -> Модуль 4".
MODULE_NAMES = ["stats__module_1", "stats__module_2", "stats__module_3", "stats__module_4"]


def mount_google_drive_if_colab():
    """Монтирует Google Drive, только если ноутбук выполняется в Colab."""
    try:
        import google.colab  # noqa: F401
        from google.colab import drive

        drive.mount("/content/drive")
        return True
    except ImportError:
        return False


def load_all_data(data_dir: str, verbose: bool = True) -> dict:
    """
    Загружает все таблицы проекта из `data_dir`.

    Parameters
    ----------
    data_dir : str
        Папка с CSV-файлами (см. TABLE_FILES).
    verbose : bool
        Печатать ли размер каждой таблицы после загрузки.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    data_dir = Path(data_dir)
    data = {}
    missing = []

    for table_name, filename in TABLE_FILES.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        df = pd.read_csv(path)
        # Приводим все колонки, оканчивающиеся на _at, к datetime —
        # это единственное безопасное системное преобразование,
        # остальную очистку (clean_id и т.п.) делаем осознанно в src/utils.py
        for col in df.columns:
            if col.endswith("_at"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        data[table_name] = df

    if missing:
        print(f"[data_loading] Не найдены файлы: {missing}. "
              f"Проверьте DATA_DIR или заполните data/README.md инструкцией по доступу.")

    if verbose:
        for name, df in data.items():
            print(f"{name:30} {len(df):>10,} строк, {df.shape[1]:>3} колонок")

    return data


def get_module_status_tables(data: dict) -> dict:
    """Возвращает только таблицы stats__module_* с добавленной колонкой `module` (int)."""
    out = {}
    for i, name in enumerate(MODULE_NAMES, start=1):
        if name in data:
            df = data[name].copy()
            df["module"] = i
            out[name] = df
    return out
