from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .common import _lower_is_better, _safe

def _aggregate_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"fold_count": len(fold_metrics), "fold_metrics": _safe(fold_metrics)}
    keys = sorted(
        {
            key
            for metrics in fold_metrics
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.number)) and value is not None
        }
    )
    for key in keys:
        values = [float(metrics[key]) for metrics in fold_metrics if metrics.get(key) is not None]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"std_{key}"] = float(np.std(values))
    return summary


def _predict_proba(model, X):
    if X is None or not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(X)
    except Exception:
        return None


def _validation_score_from_metrics(
    metrics: dict[str, Any] | None,
    optimization_metric: str | None,
) -> float | None:
    if not metrics or not optimization_metric:
        return None
    metric_key = optimization_metric.lower()
    metric_aliases = [
        metric_key,
        metric_key.removeprefix("neg_"),
        metric_key.upper(),
        metric_key.lower(),
    ]
    for alias in metric_aliases:
        for key, value in metrics.items():
            if key.lower() == alias.lower() and isinstance(value, (int, float, np.number)):
                score = float(value)
                return -score if metric_key.startswith("neg_") else score
    return None


def _best_history_score(
    history: list[dict[str, Any]],
    optimization_metric: str | None,
) -> float | None:
    scores = [row.get("validation_score") for row in history if row.get("validation_score") is not None]
    if not scores:
        return None
    if _lower_is_better(optimization_metric):
        return float(min(scores))
    return float(max(scores))
