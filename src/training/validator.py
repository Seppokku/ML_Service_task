from __future__ import annotations


def is_model_acceptable(
    metrics: dict[str, float], min_pr_auc: float, min_recall: float
) -> bool:
    return metrics.get("pr_auc", 0.0) >= min_pr_auc and metrics.get(
        "recall_at_precision", 0.0
    ) >= min_recall
