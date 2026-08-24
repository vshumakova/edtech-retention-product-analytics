from __future__ import annotations
from pathlib import Path
import pandas as pd


def export_dashboard_tables(
    output_dir: str | Path,
    funnel: pd.DataFrame,
    cohorts: pd.DataFrame,
    scored_students: pd.DataFrame,
    model_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
):
    """Export model/product outputs expected by the Streamlit app."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    funnel.to_csv(out / "dashboard_funnel.csv", index=False)
    cohorts.to_csv(out / "dashboard_cohorts.csv", index=False)
    scored_students.to_csv(out / "dashboard_students.csv", index=False)
    model_metrics.to_csv(out / "dashboard_model_metrics.csv", index=False)
    feature_importance.to_csv(out / "dashboard_feature_importance.csv", index=False)
