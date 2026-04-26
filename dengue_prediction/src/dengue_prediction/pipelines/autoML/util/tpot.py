from __future__ import annotations

import copy
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import TimeSeriesSplit

from .common import (
    _columns_with_infinite_values,
    _get_signature_parameter_names,
    _infer_task_type,
    _lower_is_better,
    _model_family,
    _pick_score,
    _records,
    _resolve_params,
    _safe,
    _safe_dict,
    _seconds_between,
    _series_has_infinite_values,
)
from .metrics import (
    _aggregate_metrics,
    _best_history_score,
    _compute_metrics,
    _predict_proba,
    _validation_score_from_metrics,
)
from .model_saving import (
    _dump_sklearn_artifact,
    _model_name,
    _unique_path,
    _write_json_artifact,
    _write_text_artifact,
    build_model_metadata,
    _model_artifact_dir,
)
from .reporting import build_tpot_report

logger = logging.getLogger(__name__)


def run_tpot_automl(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    n_splits: int = 5,
):
    config = _resolve_params(params)
    _validate_tpot_training_data(X, y)
    task_type = _infer_task_type(y, config["task_type"])
    estimator_class, tpot_params = _build_tpot_estimator(task_type, config["params"])
    optimization_metric = _resolve_tpot_optimization_metric(
        config["optimization_metric"], tpot_params, task_type
    )
    notes = [f"preset={config['preset']}"] if config["preset"] else []
    preset_name = str(config["preset"] or "default")
    fold_metrics = []

    for fold_number, (train_idx, test_idx) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(X),
        start=1,
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_started_at = time.perf_counter()
        fold_model = _make_tpot_estimator(estimator_class, tpot_params)
        _fit_tpot_estimator(fold_model, X_train, y_train)
        fold_training_time = time.perf_counter() - fold_started_at

        fold_pred = fold_model.predict(X_test)
        fold_proba = _predict_proba(fold_model, X_test)
        metrics = _compute_metrics(task_type, y_test, fold_pred, fold_proba)
        fold_metrics.append(metrics)
        save_tpot_model(
            fold_model,
            preset=preset_name,
            fold=fold_number,
            training_time=fold_training_time,
            metrics=metrics,
            validation_score=_validation_score_from_metrics(metrics, optimization_metric),
        )

    started_at = time.perf_counter()
    final_search = _make_tpot_estimator(estimator_class, tpot_params)
    _fit_tpot_estimator(final_search, X, y)
    search_time = time.perf_counter() - started_at

    final_pipeline = _get_tpot_fitted_pipeline(final_search)
    history = _tpot_history(final_search, optimization_metric)
    save_tpot_model(
        final_search,
        preset=preset_name,
        fold="final",
        training_time=search_time,
        metrics=_aggregate_metrics(fold_metrics),
        validation_score=_best_history_score(history, optimization_metric),
        save_intermediate=True,
    )
    result = build_tpot_report(
        config=config,
        task_type=task_type,
        final_model=final_pipeline,
        final_search=final_search,
        fold_metrics=fold_metrics,
        optimization_metric=optimization_metric,
        history=history,
        search_time=search_time,
        notes=notes,
        tpot_params=tpot_params,
    )
    return final_pipeline or final_search, result


def save_tpot_model(
    tpot_model: Any,
    preset: str,
    fold: int | str,
    training_time: float | None = None,
    validation_score: float | None = None,
    metrics: dict[str, Any] | None = None,
    save_intermediate: bool = False,
) -> dict[str, Any]:
    """Persist a TPOT fitted pipeline and metadata without depending on one TPOT version."""
    fold_dir = _model_artifact_dir("tpot", preset, fold)
    pipeline = _get_tpot_fitted_pipeline(tpot_model)
    pipeline_steps = _parse_pipeline_steps(pipeline)
    model_name = _model_name(pipeline or tpot_model)
    model_path = None
    save_error = None

    if pipeline is None:
        save_error = "No fitted TPOT pipeline found on fitted_pipeline_ or fitted_pipeline."
        logger.warning("Skipping TPOT model save for %s: %s", fold_dir, save_error)
    else:
        model_path = _unique_path(fold_dir / "model.joblib")
        try:
            model_path = _dump_sklearn_artifact(pipeline, model_path)
            logger.info("Saved TPOT model artifact to %s", model_path)
        except Exception as exc:
            save_error = f"Failed to serialize TPOT pipeline: {exc}"
            model_path = None
            logger.warning(save_error, exc_info=True)

        _write_text_artifact(fold_dir / "pipeline.txt", str(pipeline))
        _write_json_artifact(fold_dir / "pipeline_steps.json", pipeline_steps)

    intermediate_paths = []
    if save_intermediate:
        intermediate_paths = _save_tpot_intermediate_models(tpot_model, fold_dir)

    metadata = build_model_metadata(
        backend="TPOT",
        preset=preset,
        fold=fold,
        model_id=model_name,
        model_name=model_name,
        model_type=_model_family(pipeline or tpot_model),
        training_time=training_time,
        validation_score=validation_score,
        metrics=metrics,
        pipeline_steps=pipeline_steps,
        model_path=str(model_path) if model_path else None,
        extra={
            "pipeline_repr_path": str(fold_dir / "pipeline.txt"),
            "pipeline_steps_path": str(fold_dir / "pipeline_steps.json"),
            "intermediate_models": intermediate_paths,
            "model_save_error": save_error,
        },
    )
    _write_json_artifact(fold_dir / "metadata.json", metadata)
    logger.info("Saved TPOT model metadata to %s", fold_dir / "metadata.json")
    return metadata


def _get_tpot_fitted_pipeline(tpot_model: Any) -> Any:
    for attribute_name in ("fitted_pipeline_", "fitted_pipeline"):
        if hasattr(tpot_model, attribute_name):
            try:
                pipeline = getattr(tpot_model, attribute_name)
                if pipeline is not None:
                    return pipeline
            except Exception:
                logger.debug("Could not access TPOT %s.", attribute_name, exc_info=True)
    if hasattr(tpot_model, "steps") or hasattr(tpot_model, "predict"):
        return tpot_model
    return None


def _parse_pipeline_steps(pipeline: Any) -> dict[str, Any]:
    if pipeline is None:
        return {
            "preprocessing": [],
            "feature_transformations": [],
            "final_estimator": None,
            "all_steps": [],
        }

    steps = _pipeline_steps(pipeline)
    parsed_steps = [
        {
            "position": index,
            "name": name,
            "class_name": estimator.__class__.__name__,
            "module": estimator.__class__.__module__,
            "params": _safe_dict(_estimator_params(estimator)),
            "repr": str(estimator),
        }
        for index, (name, estimator) in enumerate(steps, start=1)
    ]
    final_estimator = parsed_steps[-1] if parsed_steps else _single_model_step(pipeline)
    intermediate = parsed_steps[:-1] if parsed_steps else []
    return {
        "preprocessing": [
            step for step in intermediate if _is_preprocessing_step(step["class_name"])
        ],
        "feature_transformations": [
            step for step in intermediate if not _is_preprocessing_step(step["class_name"])
        ],
        "final_estimator": final_estimator,
        "all_steps": parsed_steps or ([final_estimator] if final_estimator else []),
    }


def _pipeline_steps(pipeline: Any) -> list[tuple[str, Any]]:
    if hasattr(pipeline, "steps"):
        try:
            return list(getattr(pipeline, "steps"))
        except Exception:
            return []
    return []


def _single_model_step(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    return {
        "position": 1,
        "name": "model",
        "class_name": model.__class__.__name__,
        "module": model.__class__.__module__,
        "params": _safe_dict(_estimator_params(model)),
        "repr": str(model),
    }


def _estimator_params(estimator: Any) -> dict[str, Any]:
    if not hasattr(estimator, "get_params"):
        return {}
    try:
        return estimator.get_params(deep=False)
    except Exception:
        return {}


def _is_preprocessing_step(class_name: str) -> bool:
    lowered = class_name.lower()
    return any(
        token in lowered
        for token in [
            "imputer",
            "scaler",
            "encoder",
            "normalizer",
            "binarizer",
            "discretizer",
            "powertransformer",
            "quantiletransformer",
        ]
    )


def _save_tpot_intermediate_models(tpot_model: Any, fold_dir: Path) -> list[str]:
    evaluated = getattr(tpot_model, "evaluated_individuals", None)
    if not isinstance(evaluated, pd.DataFrame) or evaluated.empty:
        logger.info("No TPOT evaluated_individuals DataFrame available for intermediate saves.")
        return []

    saved_paths = []
    intermediate_dir = fold_dir / "intermediate_models"
    for model_number, (_, row) in enumerate(evaluated.iterrows(), start=1):
        candidate = row.get("Instance")
        if candidate is None or not hasattr(candidate, "predict"):
            continue
        generation = _safe(row.get("Generation"))
        generation_label = int(generation) if isinstance(generation, (int, float)) else "unknown"
        artifact_path = _unique_path(
            intermediate_dir / f"generation_{generation_label}_model_{model_number}.joblib"
        )
        try:
            saved_path = _dump_sklearn_artifact(candidate, artifact_path)
            saved_paths.append(str(saved_path))
        except Exception:
            logger.debug("Skipping TPOT intermediate model save.", exc_info=True)
    return saved_paths


def _build_tpot_estimator(task_type: str, params: dict[str, Any]) -> tuple[type[Any], dict[str, Any]]:
    try:
        from tpot import TPOTClassifier, TPOTRegressor
        from tpot.tpot_estimator.estimator import TPOTEstimator
    except Exception as exc:
        raise ImportError(f"Failed to import TPOT 1.1.0: {exc}") from exc

    estimator_class = TPOTClassifier if task_type == "classification" else TPOTRegressor
    prepared_params = _prepare_tpot_params(params, estimator_class, TPOTEstimator)
    return estimator_class, prepared_params


def _validate_tpot_training_data(X: pd.DataFrame, y: pd.Series) -> None:
    issues = []

    if X.empty:
        issues.append("feature matrix X is empty")
    if len(X) != len(y):
        issues.append(f"X and y have different row counts ({len(X)} != {len(y)})")

    duplicated_names = [str(column) for column in X.columns[X.columns.duplicated()].tolist()]
    if duplicated_names:
        issues.append("duplicated feature names: " + ", ".join(duplicated_names))

    nan_columns = [str(column) for column in X.columns[X.isna().any()].tolist()]
    if nan_columns:
        issues.append("NaN values in feature columns: " + ", ".join(nan_columns))
    if pd.Series(y).isna().any():
        issues.append("NaN values in target y")

    infinite_columns = _columns_with_infinite_values(X)
    if infinite_columns:
        issues.append("infinite values in feature columns: " + ", ".join(infinite_columns))
    if _series_has_infinite_values(pd.Series(y)):
        issues.append("infinite values in target y")

    constant_columns = [
        str(column) for column in X.columns[X.nunique(dropna=False) <= 1].tolist()
    ]
    if constant_columns:
        issues.append("constant feature columns: " + ", ".join(constant_columns))

    duplicated_value_columns = [str(column) for column in X.columns[X.T.duplicated()].tolist()]
    if duplicated_value_columns:
        issues.append(
            "duplicated feature columns by value: " + ", ".join(duplicated_value_columns)
        )

    if issues:
        raise ValueError("TPOT input validation failed before training: " + "; ".join(issues))


def _make_tpot_estimator(estimator_class: type[Any], tpot_params: dict[str, Any]):
    try:
        return estimator_class(**tpot_params)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid TPOT 1.1.0 parameters. Check conf/base/parameters.yml. "
            f"Resolved parameters were: {_safe_dict(tpot_params)}. Original error: {exc}"
        ) from exc


def _fit_tpot_estimator(estimator: Any, X: pd.DataFrame, y: pd.Series):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return estimator.fit(X, y)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TPOT failed during fit. This can happen when TPOT receives invalid "
            "parameters or incompatible training data. "
            f"Resolved parameters were: {_safe_dict(estimator.get_params())}. "
            f"Original error: {exc}"
        ) from exc


def _prepare_tpot_params(
    params: dict[str, Any],
    estimator_class: type[Any],
    estimator_base_class: type[Any],
) -> dict[str, Any]:
    prepared = copy.deepcopy(params or {})

    for old_name, new_name in {
        "mutation_rate": "mutate_probability",
        "crossover_rate": "crossover_probability",
    }.items():
        if old_name in prepared:
            if new_name in prepared and prepared[new_name] != prepared[old_name]:
                raise ValueError(
                    f"TPOT parameter conflict: '{old_name}' and '{new_name}' have different values."
                )
            prepared[new_name] = prepared.pop(old_name)

    if "cv_n_splits" in prepared:
        cv_n_splits = prepared.pop("cv_n_splits")
        if "cv" in prepared and prepared["cv"] != cv_n_splits:
            raise ValueError("TPOT parameter conflict: 'cv_n_splits' and 'cv' differ.")
        prepared["cv"] = cv_n_splits

    if "offspring_size" in prepared:
        offspring_size = prepared.pop("offspring_size")
        population_size = prepared.get("population_size")
        if population_size is None:
            prepared["population_size"] = offspring_size
        elif population_size != offspring_size:
            raise ValueError(
                "TPOT 1.1.0 no longer supports 'offspring_size' separately. "
                "Remove it or set it equal to 'population_size'."
            )

    if prepared.get("generations") is not None and prepared.get("max_time_mins") is not None:
        raise ValueError(
            "TPOT parameter conflict: configure either 'generations' or 'max_time_mins', not both."
        )
    if prepared.get("generations") is not None:
        prepared.setdefault("max_time_mins", None)

    accepted_names = _get_signature_parameter_names(
        estimator_class.__init__,
        estimator_base_class.__init__,
    )
    unknown = sorted(set(prepared) - accepted_names)
    if unknown:
        raise ValueError(
            "Unsupported TPOT 1.1.0 parameters: "
            + ", ".join(unknown)
        )

    return prepared


def _resolve_tpot_optimization_metric(
    configured_metric: str | None,
    tpot_params: dict[str, Any],
    task_type: str,
) -> str:
    if configured_metric not in {None, "auto"}:
        return str(configured_metric)

    scorers = tpot_params.get("scorers")
    if isinstance(scorers, (list, tuple)) and scorers:
        return str(scorers[0])
    if scorers:
        return str(scorers)
    if task_type == "classification":
        return "roc_auc_ovr"
    return "neg_mean_squared_error"


def _tpot_history(estimator, optimization_metric: str) -> list[dict[str, Any]]:
    history_df = getattr(estimator, "evaluated_individuals", None)
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return []

    selected_index = getattr(getattr(estimator, "selected_best_score", None), "name", None)
    score_columns = [
        f"validation_{optimization_metric}",
        optimization_metric,
        optimization_metric.lower(),
        "neg_mean_squared_error",
        "validation_neg_mean_squared_error",
        "roc_auc_ovr",
        "validation_roc_auc_ovr",
    ]

    rows = []
    for index, row in history_df.iterrows():
        model = row.get("Instance")
        rows.append(
            {
                "rank": None,
                "model_name": str(model) if model is not None else str(row.get("Individual")),
                "model_family": _model_family(model),
                "validation_score": _pick_score(row, score_columns),
                "train_score": None,
                "test_score": None,
                "fit_time": _seconds_between(row.get("Submitted Timestamp"), row.get("Completed Timestamp")),
                "predict_time": None,
                "generation_or_iteration": _safe(row.get("Generation")),
                "status": "failed" if str(row.get("Eval Error", "")).strip() not in {"", "nan", "None"} else "ok",
                "selected_in_final_ensemble": index == selected_index,
                "backend_metadata": _safe_dict(
                    {
                        "variation_function": row.get("Variation_Function"),
                        "parents": row.get("Parents"),
                        "pareto_front": row.get("Pareto_Front"),
                        "validation_pareto_front": row.get("Validation_Pareto_Front"),
                        "eval_error": row.get("Eval Error"),
                    }
                ),
            }
        )

    rows.sort(
        key=lambda item: (
            item["validation_score"] is None,
            item["validation_score"] if _lower_is_better(optimization_metric) else -item["validation_score"]
            if item["validation_score"] is not None
            else 0,
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
