from __future__ import annotations


def is_model_acceptable(
    metrics: dict[str, float], min_f1_macro: float, min_accuracy: float
) -> bool:
    return metrics.get("f1_macro", 0.0) >= min_f1_macro and metrics.get(
        "accuracy", 0.0
    ) >= min_accuracy
