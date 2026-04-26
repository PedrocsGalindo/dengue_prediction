from __future__ import annotations

import copy
import logging
import os
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from dengue_prediction.settings import DATA_DIR, PROJECT_ROOT

from .common import (
    _get_signature_parameter_names,
    _infer_task_type,
    _ms_to_seconds,
    _pick_score,
    _records,
    _resolve_params,
    _safe,
    _safe_dict,
)
from .metrics import (
    _aggregate_metrics,
    _best_history_score,
    _compute_metrics,
    _validation_score_from_metrics,
)
from .model_saving import (
    _model_artifact_dir,
    _timestamp_for_path,
    _write_json_artifact,
    build_model_metadata,
)
from .reporting import build_h2o_report

logger = logging.getLogger(__name__)


def run_h2o_automl(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    n_splits: int = 5,
):
    config = _resolve_params(params)
    X, y, validation_metadata = _prepare_h2o_training_data(X, y)
    task_type = _infer_task_type(y, config["task_type"])
    optimization_metric = config["optimization_metric"]
    notes = [f"preset={config['preset']}"] if config["preset"] else []
    preset_name = str(config["preset"] or "default")

    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError as exc:
        raise ImportError(f"Failed to import H2O AutoML: {exc}") from exc

    h2o_params, h2o_init_params, use_leaderboard_frame = _prepare_h2o_params(
        config["params"],
        H2OAutoML,
    )
    if h2o_params.get("include_algos") and h2o_params.get("exclude_algos"):
        raise ValueError("H2O AutoML does not allow include_algos and exclude_algos together.")

    h2o_params["project_name"] = h2o_params.get("project_name") or f"dengue_automl_{uuid.uuid4().hex[:8]}"
    _ensure_h2o_connection(h2o, h2o_init_params)

    target_name = y.name or "target"
    fold_metrics = []

    try:
        with _h2o_warning_context():
            for fold_number, (train_idx, test_idx) in enumerate(
                TimeSeriesSplit(n_splits=n_splits).split(X),
                start=1,
            ):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                train_h2o, test_h2o = _build_h2o_fold_frames(
                    h2o,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    target_name,
                    task_type,
                )
                x_cols = [column for column in train_h2o.columns if column != target_name]
                fold_params = copy.deepcopy(h2o_params)
                fold_params["project_name"] = f"{h2o_params['project_name']}_fold_{fold_number}"
                fold_aml = H2OAutoML(**fold_params)
                fold_started_at = time.perf_counter()
                fold_aml.train(
                    x=x_cols,
                    y=target_name,
                    training_frame=train_h2o,
                    leaderboard_frame=test_h2o if use_leaderboard_frame else None,
                )
                fold_training_time = time.perf_counter() - fold_started_at
                metrics = _h2o_metrics(fold_aml.leader, test_h2o, target_name, task_type)
                fold_metrics.append(metrics)
                save_h2o_model(
                    h2o,
                    fold_aml,
                    preset=preset_name,
                    fold=fold_number,
                    training_time=fold_training_time,
                    metrics=metrics,
                    validation_score=_validation_score_from_metrics(metrics, optimization_metric),
                    save_leaderboard_models=True,
                )

            full_h2o = _build_h2o_full_frame(h2o, X, y, target_name, task_type)
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
            model_path = _save_h2o_model(h2o, leader_model)
            save_h2o_model(
                h2o,
                aml,
                preset=preset_name,
                fold="final",
                training_time=search_time,
                metrics=_aggregate_metrics(fold_metrics),
                validation_score=_best_history_score(history, optimization_metric),
                save_leaderboard_models=True,
            )
    except Exception as exc:
        raise RuntimeError(f"H2O AutoML training failed: {exc}") from exc
    finally:
        _shutdown_h2o_cluster(h2o)

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
        notes=notes,
        h2o_params=h2o_params,
        h2o_init_params=h2o_init_params,
        leaderboard_df=leaderboard_df,
        event_log_df=event_log_df,
        aml=aml,
        validation_metadata=validation_metadata,
    )
    return model_path, result, leaderboard_df


def _build_h2o_fold_frames(
    h2o_module: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_name: str,
    task_type: str,
):
    train_df = X_train.copy()
    train_df[target_name] = y_train.values
    test_df = X_test.copy()
    test_df[target_name] = y_test.values

    train_h2o = h2o_module.H2OFrame(train_df)
    test_h2o = h2o_module.H2OFrame(test_df)
    if task_type == "classification":
        train_h2o[target_name] = train_h2o[target_name].asfactor()
        test_h2o[target_name] = test_h2o[target_name].asfactor()
    return train_h2o, test_h2o


def _build_h2o_full_frame(
    h2o_module: Any,
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    task_type: str,
):
    full_df = X.copy()
    full_df[target_name] = y.values
    full_h2o = h2o_module.H2OFrame(full_df)
    if task_type == "classification":
        full_h2o[target_name] = full_h2o[target_name].asfactor()
    return full_h2o


def save_h2o_model(
    h2o_module: Any,
    automl_or_model: Any,
    preset: str,
    fold: int | str,
    training_time: float | None = None,
    validation_score: float | None = None,
    metrics: dict[str, Any] | None = None,
    save_leaderboard_models: bool = False,
) -> dict[str, Any]:
    """Persist an H2O leader model and metadata using version-safe attribute checks."""
    fold_dir = _model_artifact_dir("h2o", preset, fold)
    leader_model = _get_h2o_leader_model(automl_or_model)
    leaderboard_df = _h2o_to_pandas(_get_h2o_leaderboard(automl_or_model))
    model_id = getattr(leader_model, "model_id", None) if leader_model is not None else None
    algo = _infer_h2o_algo(leader_model, leaderboard_df)
    model_path = None
    save_error = None

    if leader_model is None:
        save_error = "No H2O leader model found."
        logger.warning("Skipping H2O model save for %s: %s", fold_dir, save_error)
    else:
        try:
            model_path = h2o_module.save_model(leader_model, path=str(fold_dir), force=False)
            logger.info("Saved H2O leader model to %s", model_path)
        except Exception as exc:
            nested_dir = fold_dir / f"model_{_timestamp_for_path()}"
            nested_dir.mkdir(parents=True, exist_ok=True)
            try:
                model_path = h2o_module.save_model(leader_model, path=str(nested_dir), force=False)
                logger.info("Saved H2O leader model to %s", model_path)
            except Exception as nested_exc:
                save_error = f"Failed to save H2O leader model: {nested_exc}"
                logger.warning(
                    "Primary H2O save failed with %s; retry failed with %s",
                    exc,
                    nested_exc,
                    exc_info=True,
                )

    leaderboard_path = fold_dir / "leaderboard.json"
    _write_json_artifact(leaderboard_path, _records(leaderboard_df))

    intermediate_paths = []
    if save_leaderboard_models:
        intermediate_paths = _save_h2o_leaderboard_models(
            h2o_module,
            automl_or_model,
            leaderboard_df,
            fold_dir,
        )

    metadata = build_model_metadata(
        backend="H2O AutoML",
        preset=preset,
        fold=fold,
        model_id=model_id,
        model_name=model_id,
        model_type=algo,
        training_time=training_time,
        validation_score=validation_score,
        metrics=metrics,
        pipeline_steps=None,
        model_path=str(model_path) if model_path else None,
        extra={
            "algo": algo,
            "leaderboard_path": str(leaderboard_path),
            "leaderboard": _records(leaderboard_df),
            "intermediate_models": intermediate_paths,
            "model_save_error": save_error,
            "training_info": _safe_dict(getattr(automl_or_model, "training_info", {})),
        },
    )
    _write_json_artifact(fold_dir / "metadata.json", metadata)
    logger.info("Saved H2O model metadata to %s", fold_dir / "metadata.json")
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


def _save_h2o_leaderboard_models(
    h2o_module: Any,
    automl_or_model: Any,
    leaderboard_df: pd.DataFrame,
    fold_dir: Path,
) -> list[str]:
    if leaderboard_df.empty or "model_id" not in leaderboard_df.columns:
        return []
    if not hasattr(h2o_module, "get_model"):
        return []

    saved_paths = []
    leader_id = getattr(_get_h2o_leader_model(automl_or_model), "model_id", None)
    intermediate_dir = fold_dir / "leaderboard_models"
    for rank, (_, row) in enumerate(leaderboard_df.iterrows(), start=1):
        model_id = row.get("model_id")
        if not model_id or model_id == leader_id:
            continue
        algo = row.get("algo") or str(model_id).split("_", 1)[0]
        safe_algo = "".join(character if character.isalnum() else "_" for character in str(algo))
        target_dir = intermediate_dir / f"rank_{rank}_{safe_algo}"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            candidate_model = h2o_module.get_model(model_id)
            saved_path = h2o_module.save_model(candidate_model, path=str(target_dir), force=False)
            saved_paths.append(str(saved_path))
            logger.info("Saved H2O leaderboard model rank %s to %s", rank, saved_path)
        except Exception:
            logger.debug("Skipping H2O leaderboard model save for %s.", model_id, exc_info=True)
    return saved_paths


def _prepare_h2o_training_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    X_clean = pd.DataFrame(X).copy()
    y_clean = pd.Series(y).copy()

    if len(X_clean) != len(y_clean):
        raise ValueError(f"H2O input validation failed: X and y row counts differ ({len(X_clean)} != {len(y_clean)}).")

    valid_target = y_clean.notna()
    if pd.api.types.is_numeric_dtype(y_clean):
        target_values = y_clean.to_numpy(dtype=float, na_value=np.nan)
        valid_target &= np.isfinite(target_values)

    dropped_target_rows = int((~valid_target).sum())
    if dropped_target_rows:
        X_clean = X_clean.loc[valid_target].reset_index(drop=True)
        y_clean = y_clean.loc[valid_target].reset_index(drop=True)

    duplicate_names = [str(column) for column in X_clean.columns[X_clean.columns.duplicated()].tolist()]
    if duplicate_names:
        raise ValueError(
            "H2O input validation failed: duplicated feature names: "
            + ", ".join(duplicate_names)
        )

    numeric_columns = X_clean.select_dtypes(include=[np.number]).columns
    X_clean.loc[:, numeric_columns] = X_clean.loc[:, numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    filled_numeric_columns = []
    dropped_all_missing_columns = []
    for column in numeric_columns:
        if X_clean[column].isna().all():
            dropped_all_missing_columns.append(str(column))
            continue
        if X_clean[column].isna().any():
            X_clean[column] = X_clean[column].fillna(X_clean[column].median())
            filled_numeric_columns.append(str(column))

    if dropped_all_missing_columns:
        X_clean = X_clean.drop(columns=dropped_all_missing_columns)

    object_columns = X_clean.select_dtypes(exclude=[np.number]).columns
    filled_categorical_columns = []
    for column in object_columns:
        if X_clean[column].isna().any():
            X_clean[column] = X_clean[column].fillna("missing")
            filled_categorical_columns.append(str(column))

    constant_columns = [
        str(column)
        for column in X_clean.columns[X_clean.nunique(dropna=False) <= 1].tolist()
    ]
    if constant_columns:
        X_clean = X_clean.drop(columns=constant_columns)

    if X_clean.empty:
        raise ValueError("H2O input validation failed: no feature columns remain after cleaning.")
    if len(X_clean) < 2:
        raise ValueError("H2O input validation failed: at least two rows are required after cleaning.")

    return X_clean, y_clean, {
        "dropped_rows_with_invalid_target": dropped_target_rows,
        "dropped_all_missing_columns": dropped_all_missing_columns,
        "dropped_constant_columns": constant_columns,
        "filled_numeric_columns": filled_numeric_columns,
        "filled_categorical_columns": filled_categorical_columns,
        "row_count": len(X_clean),
        "feature_count": len(X_clean.columns),
    }


@contextmanager
def _h2o_warning_context():
    try:
        from h2o.exceptions import H2ODependencyWarning
    except Exception:
        H2ODependencyWarning = None

    with warnings.catch_warnings():
        if H2ODependencyWarning is not None:
            warnings.filterwarnings("ignore", category=H2ODependencyWarning)
        yield


def _save_h2o_model(h2o_module: Any, model: Any) -> str:
    model_dir = DATA_DIR / "06_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        return str(h2o_module.save_model(model, path=str(model_dir), force=True))
    except Exception as exc:
        raise RuntimeError(f"Failed to save H2O model to {model_dir}: {exc}") from exc


def _shutdown_h2o_cluster(h2o_module: Any) -> None:
    try:
        h2o_module.cluster().shutdown(prompt=False)
    except Exception:
        logger.debug("H2O shutdown skipped or failed.", exc_info=True)


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


def _ensure_h2o_connection(h2o_module: Any, init_params: dict[str, Any]) -> None:
    try:
        connection = h2o_module.connection()
        if connection is not None:
            return
    except Exception:
        pass

    resolved_init_params = copy.deepcopy(init_params or {})
    if "ice_root" not in resolved_init_params:
        ice_root = PROJECT_ROOT / "temp" / "h2o_runtime"
        ice_root.mkdir(parents=True, exist_ok=True)
        resolved_init_params["ice_root"] = str(ice_root)
    if "log_dir" not in resolved_init_params:
        resolved_init_params["log_dir"] = resolved_init_params["ice_root"]
    resolved_init_params.setdefault("verbose", False)

    temp_keys = ("TMP", "TEMP", "TMPDIR")
    previous_temp_env = {key: os.environ.get(key) for key in temp_keys}
    previous_tempdir = tempfile.tempdir
    previous_mkdtemp = tempfile.mkdtemp

    def _workspace_mkdtemp(*args, **kwargs):
        temp_root = Path(resolved_init_params["ice_root"])
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_root / f"tmp_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        return str(temp_dir)

    for key in temp_keys:
        os.environ[key] = str(resolved_init_params["ice_root"])
    tempfile.tempdir = str(resolved_init_params["ice_root"])
    tempfile.mkdtemp = _workspace_mkdtemp

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h2o_module.init(**resolved_init_params)
    except Exception as exc:
        details = ", ".join(
            f"{key}={value!r}" for key, value in sorted(resolved_init_params.items())
        )
        raise RuntimeError(
            "Failed to initialize H2O. Ensure Java is installed and that H2O can write to a "
            f"local temp directory. h2o.init({details}) failed with: {exc}"
        ) from exc
    finally:
        tempfile.tempdir = previous_tempdir
        tempfile.mkdtemp = previous_mkdtemp
        for key, value in previous_temp_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    if task_type == "classification":
        y_pred = predictions.iloc[:, 0]
        y_proba = predictions.iloc[:, 1:] if predictions.shape[1] > 1 else None
        return _compute_metrics(task_type, y_true, y_pred, y_proba)
    return _compute_metrics(task_type, y_true, predictions.iloc[:, 0])


def _h2o_to_pandas(frame: Any) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    try:
        return frame.as_data_frame()
    except Exception:
        return pd.DataFrame()
