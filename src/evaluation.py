from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


def ranking_metrics(y_true, score, k_share=0.20):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score)
    n = max(1, int(np.ceil(len(y) * k_share)))
    idx = np.argsort(-s)[:n]

    base_rate = y.mean()
    precision_k = y[idx].mean()
    recall_k = y[idx].sum() / max(y.sum(), 1)
    lift_k = precision_k / base_rate if base_rate > 0 else np.nan

    return {
        "k_share": k_share,
        "contacted": n,
        "precision_at_k": precision_k,
        "recall_at_k": recall_k,
        "lift_at_k": lift_k,
    }


def evaluate_scores(y_true, score, k_share=0.20):
    out = {
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "brier": brier_score_loss(y_true, score),
    }
    out.update(ranking_metrics(y_true, score, k_share))
    return out


def gains_table(y_true, score, bins=10):
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(score)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index + 1, q=bins, labels=False, duplicates="drop") + 1
    out = (
        df.groupby("decile")
        .agg(students=("y", "size"), churners=("y", "sum"), churn_rate=("y", "mean"))
        .reset_index()
    )
    base = df["y"].mean()
    out["lift"] = out["churn_rate"] / base if base > 0 else np.nan
    return out
