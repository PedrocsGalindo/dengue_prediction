from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tpot import TPOTRegressor

from dengue_prediction.settings import DATA_DIR

from .reports import save_automl_outputs


def run_tpot_automl(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None,
):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = DATA_DIR / "results" / "autoML" / "tpot" / run_id
    init_params, preset_params, metric, preset = _prepare_params(params)
    print("\n\n")
    print(init_params)
    print(preset_params)
    print("\n\n")
    fold_metrics = []

    started_at = time.perf_counter()
    search = TPOTRegressor(
        search_space=init_params.get("search_space"),
        scorers=init_params.get("scorers"),
        scorers_weights=init_params.get("scorers_weights"),
        preprocessing=init_params.get("preprocessing"),
        validation_strategy=init_params.get("validation_strategy"),
        verbose=init_params.get("verbose"),
        random_state=init_params.get("random_state"),

        generations=preset_params.get("generations"),
        population_size=preset_params.get("population_size"),
        max_eval_time_mins=preset_params.get("max_eval_time_mins"),
        early_stop=preset_params.get("early_stop"),
        cv=preset_params.get("cv"),
        n_jobs=preset_params.get("n_jobs"),
        processes=preset_params.get("processes"),
        mutate_probability=preset_params.get("mutate_probability"),
        crossover_probability=preset_params.get("crossover_probability"),
    )
    search.fit(X, y)
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
            "resolved_params": _safe(preset_params),
            "training_time_seconds": float(training_time),
            "output_dir": str(output_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    save_automl_outputs(result, output_dir)
    return best_pipeline, result

def _prepare_params(
    params: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], str, str | None]:
    params = dict(params or {})

    presets = params.pop("presets", {}) or {}
    preset = params.pop("preset", None)
    metric = str(params.pop("optimization_metric", "neg_mean_squared_error"))

    # Não é parâmetro do TPOTRegressor
    params.pop("task_type", None)

    if preset not in presets:
        raise ValueError(f"Unknown TPOT preset: {preset}")

    # Parâmetros que são padrão para todos os presets
    init_params = {
        key: value
        for key, value in params.items()
        if value is not None
    }

    # Parâmetros específicos do preset escolhido: low, medium ou high
    preset_params = {
        key: value
        for key, value in presets[preset].items()
        if value is not None
    }

    return init_params, preset_params, metric, preset

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
