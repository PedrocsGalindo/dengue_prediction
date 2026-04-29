from __future__ import annotations

import copy
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import math

from dengue_prediction.settings import PROJECT_ROOT

from .common import (
    _get_signature_parameter_names,
    _ms_to_seconds,
    _pick_score,
    _resolve_params,
    _safe,
    _safe_dict,
)
from .metrics import (
    _aggregate_metrics,
    _best_history_score,
)
from .model_saving import (
    build_model_metadata,
    save_model_artifacts,
)
from .reporting import build_h2o_report

import h2o
from h2o.automl import H2OAutoML

logger = logging.getLogger(__name__)


def run_h2o_automl(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    n_splits: int = 5,
):
    config = _resolve_params(params)
    task_type = "regression"
    optimization_metric = config["optimization_metric"]
    preset_name = config["preset"]
    target_name = "casos_dengue"
    full_df = X.copy()
    full_df[target_name] = y.values

    h2o.init()


    h2o_params, h2o_init_params, use_leaderboard_frame = _prepare_h2o_params(
        config["params"],
        H2OAutoML,
    )
    if h2o_params.get("include_algos") and h2o_params.get("exclude_algos"):
        raise ValueError("H2O AutoML does not allow include_algos and exclude_algos together.")

    h2o_params["project_name"] = h2o_params.get("project_name") or f"dengue_automl_{uuid.uuid4().hex[:8]}"

    fold_metrics = []

    try:
        for fold_number, (train_idx, test_idx) in enumerate(
            TimeSeriesSplit(n_splits=n_splits).split(full_df),
            start=1,
        ):
            train_df = full_df.iloc[train_idx]
            test_df = full_df.iloc[test_idx]

            train_h2o = h2o.H2OFrame(train_df)
            test_h2o = h2o.H2OFrame(test_df)
            x_cols = [column for column in train_h2o.columns if column != target_name]
            fold_params = copy.deepcopy(h2o_params)
            fold_params["project_name"] = f"{h2o_params['project_name']}_fold_{fold_number}"
            fold_aml = H2OAutoML(**fold_params)
            fold_aml.train(
                x=x_cols,
                y=target_name,
                training_frame=train_h2o,
                leaderboard_frame=test_h2o if use_leaderboard_frame else None,
            )
            metrics = _h2o_metrics(fold_aml.leader, test_h2o, target_name, task_type)
            fold_metrics.append(metrics)

        full_h2o = _build_h2o_full_frame(X, y, target_name)
        x_cols = [column for column in full_h2o.columns if column != target_name]

        started_at = time.perf_counter()
        aml = H2OAutoML(**h2o_params)
        aml.train(x=x_cols, y=target_name, training_frame=full_h2o)
        search_time = time.perf_counter() - started_at

        leader_model = aml.leader
        if leader_model is None:
            raise RuntimeError("H2O AutoML completed without a leader model.")
        best_model_id = getattr(leader_model, "model_id", "unknown")

        leaderboard_df = _h2o_to_pandas(_get_h2o_leaderboard(aml))
        event_log_df = _h2o_to_pandas(getattr(aml, "event_log", None))
        history = _h2o_history(aml, optimization_metric)
        model_metadata = save_h2o_model(
            h2o,
            aml,
            preset=preset_name,
            fold="final",
            training_time=search_time,
            metrics=_aggregate_metrics(fold_metrics),
            validation_score=_best_history_score(history, optimization_metric),
        )
        model_path = model_metadata.get("model_path")
        
    except Exception as exc:
        raise RuntimeError(f"H2O AutoML training failed: {exc}") from exc
    h2o.cluster().shutdown(prompt=False)
    logger.info("Saved H2O best model to %s", model_path)
    logger.info("H2O best model: %s", best_model_id)
    logger.info("H2O training duration: %.2f seconds", search_time)

    result = build_h2o_report(
        config=config,
        task_type=task_type,
        best_model_id=best_model_id,
        model_path=model_path,
        fold_metrics=fold_metrics,
        optimization_metric=optimization_metric,
        history=history,
        search_time=search_time,
        notes=[f"preset={preset_name}"],
        h2o_params=h2o_params,
        h2o_init_params=h2o_init_params,
        leaderboard_df=leaderboard_df,
        event_log_df=event_log_df,
        aml=aml
    )
    result["saved_model"] = model_metadata
    result.setdefault("backend_artifacts", {})["model_path"] = model_path
    return model_path, result

def _build_h2o_full_frame(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
):
    full_df = X.copy()
    full_df[target_name] = y.values
    full_h2o = h2o.H2OFrame(full_df)
    return full_h2o


def save_h2o_model(
    h2o: Any,
    automl_or_model: Any,
    preset: str,
    fold: int | str,
    training_time: float | None = None,
    validation_score: float | None = None,
    metrics: dict[str, Any] | None = None,
    save_leaderboard_models: bool = False,
) -> dict[str, Any]:
    """Persist only the H2O leader model using version-safe attribute checks."""
    leader_model = _get_h2o_leader_model(automl_or_model)
    leaderboard_df = _h2o_to_pandas(_get_h2o_leaderboard(automl_or_model))
    model_id = getattr(leader_model, "model_id", None) if leader_model is not None else None
    algo = _infer_h2o_algo(leader_model, leaderboard_df)

    if leader_model is None:
        logger.warning("Skipping H2O model save: no leader model found.")
        return build_model_metadata(
            backend="H2O AutoML",
            preset=preset,
            fold=fold,
            model_id=None,
            model_name=None,
            model_type=algo,
            training_time=training_time,
            validation_score=validation_score,
            metrics=metrics,
            pipeline_steps=None,
            model_path=None,
            extra={
                "algo": algo,
                "model_save_error": "No H2O leader model found.",
            },
        )

    if save_leaderboard_models:
        logger.info(
            "Ignoring save_leaderboard_models=True; only the H2O leader model is persisted."
        )

    metadata = save_model_artifacts(
        backend="H2O AutoML",
        preset=preset,
        model=leader_model,
        fold=fold,
        model_id=model_id or "h2o_leader",
        model_name=model_id,
        model_type=algo,
        training_time=training_time,
        validation_score=validation_score,
        metrics=metrics,
        pipeline_steps=None,
        h2o_module=h2o,
        extra={
            "algo": algo,
            "training_info": _safe_dict(getattr(automl_or_model, "training_info", {})),
        },
    )
    logger.info("Saved H2O leader model to %s", metadata.get("model_path"))
    return metadata


def _get_h2o_leader_model(automl_or_model: Any) -> Any:
    if hasattr(automl_or_model, "leader"):
        try:
            leader = getattr(automl_or_model, "leader")
            if leader is not None:
                return leader
        except Exception:
            logger.debug("Could not access H2O AutoML leader.", exc_info=True)
    if hasattr(automl_or_model, "model_id"):
        return automl_or_model
    return None


def _infer_h2o_algo(model: Any, leaderboard_df: pd.DataFrame | None = None) -> str | None:
    model_id = getattr(model, "model_id", None) if model is not None else None
    if leaderboard_df is not None and not leaderboard_df.empty and model_id:
        try:
            matching_rows = leaderboard_df[leaderboard_df["model_id"] == model_id]
            if not matching_rows.empty and "algo" in matching_rows.columns:
                return _safe(matching_rows.iloc[0].get("algo"))
        except Exception:
            logger.debug("Could not infer H2O algo from leaderboard.", exc_info=True)
    for attribute_name in ("algo", "_model_json"):
        if hasattr(model, attribute_name):
            try:
                value = getattr(model, attribute_name)
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    output = value.get("output", {})
                    algo = output.get("algo") or value.get("algo")
                    if algo:
                        return str(algo)
            except Exception:
                continue
    if model_id and "_" in model_id:
        return model_id.split("_", 1)[0]
    return model.__class__.__name__ if model is not None else None
def _prepare_h2o_params(
    params: dict[str, Any],
    estimator_class: type[Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    prepared = copy.deepcopy(params or {})
    init_params = prepared.pop("init", {}) or {}
    use_leaderboard_frame = bool(prepared.pop("use_leaderboard_frame", True))

    if not isinstance(init_params, dict):
        raise ValueError("H2O 'init' parameters must be provided as a dictionary.")

    accepted_names = _get_signature_parameter_names(estimator_class.__init__)
    unknown = sorted(set(prepared) - accepted_names)
    if unknown:
        raise ValueError(
            "Unsupported H2O AutoML parameters for h2o==3.46.0.10: "
            + ", ".join(unknown)
        )

    return prepared, init_params, use_leaderboard_frame

def _get_h2o_leaderboard(aml: Any):
    try:
        from h2o.automl import get_leaderboard

        return get_leaderboard(aml, extra_columns="ALL")
    except Exception:
        return getattr(aml, "leaderboard", None)


def _h2o_history(aml, optimization_metric: str) -> list[dict[str, Any]]:
    leaderboard_df = _h2o_to_pandas(_get_h2o_leaderboard(aml))
    if leaderboard_df.empty:
        return []

    leader_id = getattr(getattr(aml, "leader", None), "model_id", None)
    rows = []
    for rank, (_, row) in enumerate(leaderboard_df.iterrows(), start=1):
        rows.append(
            {
                "rank": rank,
                "model_name": row.get("model_id"),
                "model_family": row.get("algo"),
                "validation_score": _pick_score(
                    row,
                    [
                        optimization_metric,
                        optimization_metric.lower(),
                        "RMSE",
                        "rmse",
                        "MAE",
                        "mae",
                        "MSE",
                        "mse",
                        "r2",
                        "AUC",
                        "auc",
                        "logloss",
                    ],
                ),
                "train_score": None,
                "test_score": None,
                "fit_time": _ms_to_seconds(row.get("training_time_ms")),
                "predict_time": _ms_to_seconds(row.get("predict_time_per_row_ms")),
                "generation_or_iteration": rank,
                "status": "ok",
                "selected_in_final_ensemble": row.get("model_id") == leader_id,
                "backend_metadata": _safe_dict(row.to_dict()),
            }
        )
    return rows


def _h2o_metrics(model, test_frame, target_name: str, task_type: str) -> dict[str, Any]:
    y_true = _h2o_to_pandas(test_frame[target_name]).iloc[:, 0]
    predictions = _h2o_to_pandas(model.predict(test_frame))
    mse = mean_squared_error(y_true, predictions)
    return {
            "mae": float(mean_absolute_error(y_true, predictions)),
            "mse": float(mse),
            "rmse": float(math.sqrt(mse)),
            "r2": float(r2_score(y_true, predictions)),
        }


def _h2o_to_pandas(frame: Any) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    try:
        return frame.as_data_frame()
    except Exception:
        return pd.DataFrame()
