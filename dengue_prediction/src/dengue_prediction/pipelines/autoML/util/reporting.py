from __future__ import annotations

from typing import Any

import pandas as pd

from .common import (
    _call_if_exists,
    _get_attr_or_default,
    _lower_is_better,
    _pick_score,
    _records,
    _safe,
    _safe_dict,
)
from .metrics import _aggregate_metrics
from .model_saving import save_experiment_results

def save_automl_report(
    report: dict[str, Any],
    params: dict[str, Any],
    framework_name: str,
) -> str:
    return save_experiment_results(
        report=report,
        params=params,
        framework_name=framework_name,
    )


def _search_summary(
    history: list[dict[str, Any]],
    search_time: float,
    optimization_metric: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    scores = [row["validation_score"] for row in history if row.get("validation_score") is not None]
    if not scores:
        best_score = None
    elif _lower_is_better(optimization_metric):
        best_score = min(scores)
    else:
        best_score = max(scores)

    return {
        "total_models_evaluated": len(history),
        "successful_models": sum(row.get("status") == "ok" for row in history),
        "failed_models": sum(row.get("status") != "ok" for row in history),
        "search_time": float(search_time),
        "best_score_seen": _safe(best_score),
        "notes": [note for note in (notes or []) if note],
    }


def _autosklearn_history(estimator, optimization_metric: str) -> list[dict[str, Any]]:
    leaderboard_df = pd.DataFrame()
    if hasattr(estimator, "leaderboard"):
        try:
            leaderboard_df = estimator.leaderboard(detailed=True)
        except TypeError:
            try:
                leaderboard_df = estimator.leaderboard()
            except Exception:
                leaderboard_df = pd.DataFrame()
        except Exception:
            leaderboard_df = pd.DataFrame()

    if not leaderboard_df.empty:
        rows = []
        for rank, (_, row) in enumerate(leaderboard_df.iterrows(), start=1):
            model_name = (
                row.get("model_id")
                or row.get("name")
                or row.get("type")
                or row.get("classifier")
                or row.get("regressor")
                or f"candidate_{rank}"
            )
            rows.append(
                {
                    "rank": int(row.get("rank", rank)),
                    "model_name": str(model_name),
                    "model_family": str(row.get("type", model_name)),
                    "validation_score": _pick_score(
                        row,
                        [optimization_metric, "score", "mean_test_score", "cost"],
                    ),
                    "train_score": _safe(row.get("mean_train_score")),
                    "test_score": None,
                    "fit_time": _safe(row.get("mean_fit_time")),
                    "predict_time": _safe(row.get("mean_score_time")),
                    "generation_or_iteration": rank,
                    "status": str(row.get("status", "ok")),
                    "selected_in_final_ensemble": False,
                    "backend_metadata": _safe_dict(row.to_dict()),
                }
            )
        return rows

    cv_results = pd.DataFrame(_get_attr_or_default(estimator, "cv_results_", {}))
    if cv_results.empty:
        return []

    rows = []
    for rank, (_, row) in enumerate(cv_results.iterrows(), start=1):
        params = row.get("params", {})
        model_family = (
            params.get("regressor:__choice__")
            or params.get("classifier:__choice__")
            or params.get("feature_preprocessor:__choice__")
            or "auto-sklearn-candidate"
        )
        rows.append(
            {
                "rank": int(row.get("rank_test_scores", rank)),
                "model_name": str(params),
                "model_family": str(model_family),
                "validation_score": _pick_score(row, ["mean_test_score", optimization_metric]),
                "train_score": _safe(row.get("mean_train_score")),
                "test_score": None,
                "fit_time": _safe(row.get("mean_fit_time")),
                "predict_time": _safe(row.get("mean_score_time")),
                "generation_or_iteration": rank,
                "status": str(row.get("status", "ok")),
                "selected_in_final_ensemble": False,
                "backend_metadata": _safe_dict(row.to_dict()),
            }
        )
    return rows


def build_tpot_report(
    config: dict[str, Any],
    task_type: str,
    final_model: Any,
    final_search: Any,
    fold_metrics: list[dict[str, Any]],
    optimization_metric: str,
    history: list[dict[str, Any]],
    search_time: float,
    notes: list[str],
    tpot_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "backend": "TPOT",
        "preset": config["preset"],
        "task_type": task_type,
        "final_model_repr": str(final_model or final_search),
        "final_model_name": str(final_model or final_search),
        "final_metrics": _aggregate_metrics(fold_metrics),
        "optimization_metric": optimization_metric,
        "model_history": history,
        "search_summary": _search_summary(history, search_time, optimization_metric, notes),
        "backend_artifacts": {
            "resolved_params": _safe_dict(tpot_params),
            "selected_best_score": _safe(getattr(final_search, "selected_best_score", None)),
            "pareto_front": _records(getattr(final_search, "pareto_front", None)),
            "fold_metrics": _safe(fold_metrics),
        },
    }


def build_h2o_report(
    config: dict[str, Any],
    task_type: str,
    best_model_id: str,
    model_path: str,
    fold_metrics: list[dict[str, Any]],
    optimization_metric: str,
    history: list[dict[str, Any]],
    search_time: float,
    notes: list[str],
    h2o_params: dict[str, Any],
    h2o_init_params: dict[str, Any],
    leaderboard_df: pd.DataFrame,
    event_log_df: pd.DataFrame,
    aml: Any,
    validation_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "backend": "H2O AutoML",
        "preset": config["preset"],
        "task_type": task_type,
        "final_model_repr": best_model_id,
        "final_model_name": best_model_id,
        "final_metrics": _aggregate_metrics(fold_metrics),
        "optimization_metric": optimization_metric,
        "model_history": history,
        "search_summary": _search_summary(history, search_time, optimization_metric, notes),
        "backend_artifacts": {
            "model_path": model_path,
            "resolved_params": _safe_dict(h2o_params),
            "leaderboard": _records(leaderboard_df),
            "event_log": _records(event_log_df),
            "training_info": _safe_dict(getattr(aml, "training_info", {})),
            "fold_metrics": _safe(fold_metrics),
            "init_params": _safe_dict(h2o_init_params),
            "validation": _safe_dict(validation_metadata),
        },
    }


def build_autosklearn_report(
    config: dict[str, Any],
    task_type: str,
    automl: Any,
    fold_metrics: list[dict[str, Any]],
    optimization_metric: str,
    history: list[dict[str, Any]],
    search_time: float,
    notes: list[str],
    automl_params: dict[str, Any],
    leaderboard_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "backend": "auto-sklearn",
        "preset": config["preset"],
        "task_type": task_type,
        "final_model_repr": automl.__class__.__name__,
        "final_model_name": automl.__class__.__name__,
        "final_metrics": _aggregate_metrics(fold_metrics),
        "optimization_metric": optimization_metric,
        "model_history": history,
        "search_summary": _search_summary(history, search_time, optimization_metric, notes),
        "backend_artifacts": {
            "resolved_params": _safe_dict(automl_params),
            "leaderboard": _records(leaderboard_df),
            "show_models": _safe(_call_if_exists(automl, "show_models")),
            "performance_over_time": _records(
                _get_attr_or_default(automl, "performance_over_time_", pd.DataFrame())
            ),
            "sprint_statistics": _safe(_call_if_exists(automl, "sprint_statistics")),
            "cv_results": _safe_dict(_get_attr_or_default(automl, "cv_results_", {})),
            "fold_metrics": _safe(fold_metrics),
        },
    }
