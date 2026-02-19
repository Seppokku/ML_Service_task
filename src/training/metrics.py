from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    precision_threshold: float = 0.6,
    top_k_ratio: float = 0.1,
) -> dict[str, float]:
    pr_auc = float(average_precision_score(y_true, y_prob))

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    recall_at_precision = 0.0
    for p, r in zip(precision, recall):
        if p >= precision_threshold:
            recall_at_precision = max(recall_at_precision, float(r))

    best_f1 = 0.0
    best_f1_threshold = 0.5
    best_f2 = 0.0
    best_f2_threshold = 0.5
    best_f2_precision = 0.0
    best_f2_recall = 0.0
    if thresholds.size > 0:
        pr_for_thresholds = precision[:-1]
        rc_for_thresholds = recall[:-1]
        denom_f1 = pr_for_thresholds + rc_for_thresholds + 1e-12
        f1_scores = 2 * (pr_for_thresholds * rc_for_thresholds) / denom_f1
        f1_idx = int(np.argmax(f1_scores))
        best_f1 = float(f1_scores[f1_idx])
        best_f1_threshold = float(thresholds[f1_idx])

        beta = 2.0
        denom_f2 = (beta * beta * pr_for_thresholds) + rc_for_thresholds + 1e-12
        f2_scores = (1 + beta * beta) * (pr_for_thresholds * rc_for_thresholds) / denom_f2
        f2_idx = int(np.argmax(f2_scores))
        best_f2 = float(f2_scores[f2_idx])
        best_f2_threshold = float(thresholds[f2_idx])
        best_f2_precision = float(pr_for_thresholds[f2_idx])
        best_f2_recall = float(rc_for_thresholds[f2_idx])

    top_k = max(1, int(len(y_true) * top_k_ratio))
    top_idx = np.argsort(y_prob)[::-1][:top_k]
    positives = y_true.sum()
    recall_at_top_k = float(y_true[top_idx].sum() / positives) if positives > 0 else 0.0

    return {
        "pr_auc": pr_auc,
        "recall_at_precision": recall_at_precision,
        "precision_threshold": float(precision_threshold),
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_threshold,
        "best_f2": best_f2,
        "best_f2_threshold": best_f2_threshold,
        "best_f2_precision": best_f2_precision,
        "best_f2_recall": best_f2_recall,
        "recall_at_top_k": recall_at_top_k,
        "top_k_ratio": float(top_k_ratio),
    }
