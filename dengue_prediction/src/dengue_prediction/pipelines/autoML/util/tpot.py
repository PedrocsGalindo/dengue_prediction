from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tpot import TPOTRegressor

from dengue_prediction.settings import DATA_DIR

from ..reports import save_automl_outputs


def run_tpot_automl(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None,
    n_splits: int = 5,
):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = DATA_DIR / "results" / "autoML" / "tpot" / run_id
    X_train, y_train, dropped_columns = _prepare_data(X, y)
    tpot_params, metric, preset = _prepare_params(params)

    fold_metrics = []
    if n_splits >= 2:
        for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X_train):
            model = TPOTRegressor(
                search_space=tpot_params.get("search_space"),
                scorers=tpot_params.get("scorers"),
                scorers_weights=tpot_params.get("scorers_weights"),
                preprocessing=tpot_params.get("preprocessing"),
                validation_strategy=tpot_params.get("validation_strategy"),
                verbose=tpot_params.get("verbose"),
                random_state=tpot_params.get("random_state"),
                generations=tpot_params.get("generations"),
                population_size=tpot_params.get("population_size"),
                max_eval_time_mins=tpot_params.get("max_eval_time_mins"),
                early_stop=tpot_params.get("early_stop"),
                cv=tpot_params.get("cv"),
                n_jobs=tpot_params.get("n_jobs"),
                processes=tpot_params.get("processes"),
                mutate_probability=tpot_params.get("mutate_probability"),
                crossover_probability=tpot_params.get("crossover_probability"),
            )
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            predictions = model.predict(X_train.iloc[test_idx])
            fold_metrics.append(_regression_metrics(y_train.iloc[test_idx], predictions))

    mean_metrics = {}
    if fold_metrics:
        mean_metrics = {
            metric_name: float(np.mean([fold[metric_name] for fold in fold_metrics]))
            for metric_name in fold_metrics[0]
        }

    started_at = time.perf_counter()
    search = TPOTRegressor(
        search_space=tpot_params.get("search_space"),
        scorers=tpot_params.get("scorers"),
        scorers_weights=tpot_params.get("scorers_weights"),
        preprocessing=tpot_params.get("preprocessing"),
        validation_strategy=tpot_params.get("validation_strategy"),
        verbose=tpot_params.get("verbose"),
        random_state=tpot_params.get("random_state"),
        generations=tpot_params.get("generations"),
        population_size=tpot_params.get("population_size"),
        max_eval_time_mins=tpot_params.get("max_eval_time_mins"),
        early_stop=tpot_params.get("early_stop"),
        cv=tpot_params.get("cv"),
        n_jobs=tpot_params.get("n_jobs"),
        processes=tpot_params.get("processes"),
        mutate_probability=tpot_params.get("mutate_probability"),
        crossover_probability=tpot_params.get("crossover_probability"),
    )
    search.fit(X_train, y_train)
    training_time = time.perf_counter() - started_at

    best_pipeline = (
        getattr(search, "fitted_pipeline_", None)
        or getattr(search, "fitted_pipeline", None)
        or search
    )
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "best_model.joblib"
    pipeline_path = artifact_dir / "pipeline.txt"
    joblib.dump(best_pipeline, model_path)
    pipeline_path.write_text(str(best_pipeline), encoding="utf-8")

    result = {
        "search_history": _search_history(search, metric),
        "best_model": {
            "model_name": str(best_pipeline),
            "model_family": _model_family(best_pipeline),
            "hyperparameters": _params(best_pipeline),
            "score": {
                "metric": metric,
                "value": _safe(getattr(search, "selected_best_score", None)),
                "source": "selected_best_score",
            },
            "artifact_path": str(model_path),
            "artifacts": {
                "artifact_path": str(model_path),
                "pipeline_repr_path": str(pipeline_path),
            },
        },
        "metadata": {
            "backend": "TPOT",
            "task_type": "regression",
            "run_id": run_id,
            "preset": preset,
            "optimization_metric": metric,
            "resolved_params": _safe(tpot_params),
            "row_count": int(len(X_train)),
            "feature_count": int(X_train.shape[1]),
            "dropped_columns": dropped_columns,
            "n_splits": int(n_splits),
            "training_time_seconds": float(training_time),
            "evaluation": {
                "fold_metrics": fold_metrics,
                "mean_metrics": mean_metrics,
            },
            "output_dir": str(output_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    save_automl_outputs(result, output_dir)
    return best_pipeline, result


def _prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    X_clean = pd.DataFrame(X).copy()
    y_clean = pd.Series(y).copy()
    dropped_columns = [column for column in ["data"] if column in X_clean.columns]
    X_clean = X_clean.drop(columns=dropped_columns, errors="ignore")

    if X_clean.empty:
        raise ValueError("TPOT training failed: X has no feature columns.")
    if len(X_clean) != len(y_clean):
        raise ValueError("TPOT training failed: X and y have different lengths.")
    if y_clean.isna().any():
        raise ValueError("TPOT training failed: target y contains missing values.")
    if not pd.api.types.is_numeric_dtype(y_clean):
        raise ValueError("TPOT training failed: target y must be numeric.")

    numeric_X = X_clean.select_dtypes(include=[np.number])
    if np.isinf(numeric_X.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("TPOT training failed: X contains infinite numeric values.")
    if np.isinf(y_clean.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("TPOT training failed: target y contains infinite values.")

    return X_clean.reset_index(drop=True), y_clean.reset_index(drop=True), dropped_columns


def _prepare_params(params: dict[str, Any] | None) -> tuple[dict[str, Any], str, str | None]:
    raw = dict(params or {})
    presets = raw.pop("presets", {}) or {}
    preset = raw.pop("preset", None)
    overrides = raw.pop("overrides", {}) or {}
    raw.pop("task_type", None)
    metric = str(raw.pop("optimization_metric", "neg_mean_squared_error"))

    if preset and presets and preset not in presets:
        raise ValueError(f"Unknown TPOT preset: {preset}")

    resolved = dict(presets.get(preset, {}))
    resolved.update({key: value for key, value in raw.items() if value is not None})
    resolved.update({key: value for key, value in overrides.items() if value is not None})
    return resolved, metric, preset


def _regression_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _search_history(search: Any, metric: str) -> list[dict[str, Any]]:
    history = getattr(search, "evaluated_individuals", None)
    if not isinstance(history, pd.DataFrame) or history.empty:
        return []

    score_columns = [
        f"validation_{metric}",
        metric,
        metric.lower(),
        "validation_neg_mean_squared_error",
        "neg_mean_squared_error",
    ]
    rows = []
    for iteration, (_, row) in enumerate(history.iterrows(), start=1):
        model = row.get("Instance")
        score_column = next((column for column in score_columns if column in row.index), None)
        rows.append(
            {
                "iteration": iteration,
                "generation": _safe(row.get("Generation")),
                "rank": _safe(row.get("Rank")),
                "model_name": str(model if model is not None else row.get("Individual")),
                "model_family": _model_family(model),
                "hyperparameters": _params(model),
                "score": {
                    "metric": metric,
                    "value": _safe(row.get(score_column)) if score_column else None,
                    "source_column": score_column,
                },
                "status": "failed"
                if str(row.get("Eval Error", "")).strip() not in {"", "nan", "None"}
                else "ok",
                "extra": {"raw_tpot_row": _safe(row.to_dict())},
            }
        )
    return rows


def _model_family(model: Any) -> str | None:
    if model is None:
        return None
    steps = getattr(model, "steps", None)
    return steps[-1][1].__class__.__name__ if steps else model.__class__.__name__


def _params(model: Any) -> dict[str, Any]:
    if model is None or not hasattr(model, "get_params"):
        return {}
    try:
        return _safe(model.get_params(deep=True))
    except Exception:
        return {}


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
