from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_prediction.settings import DATA_DIR

from .common import _lower_is_better, _resolve_params, _safe, _safe_dict

logger = logging.getLogger(__name__)


AUTOML_RESULTS_DIRNAME = "AutoML"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"


def automl_results_dir(preset: str | None) -> Path:
    path = DATA_DIR / "results" / AUTOML_RESULTS_DIRNAME / str(preset or "default")
    path.mkdir(parents=True, exist_ok=True)
    return path


def automl_models_dir(preset: str | None) -> Path:
    path = DATA_DIR / "results" / "models" / AUTOML_RESULTS_DIRNAME / str(preset or "default")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model_metadata(
    backend: str,
    preset: str,
    model_id: str | None,
    model_name: str | None,
    model_type: str | None,
    training_time: float | None,
    validation_score: float | None,
    metrics: dict[str, Any] | None,
    pipeline_steps: dict[str, Any] | None,
    model_path: str | None,
    fold: int | str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "backend": backend,
        "preset": preset,
        "model_id": model_id,
        "model_name": model_name,
        "model_type": model_type,
        "training_time": _safe(training_time),
        "validation_score": _safe(validation_score),
        "metrics": _safe_dict(metrics),
        "pipeline_steps": _safe(pipeline_steps),
        "model_path": model_path,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    if fold is not None:
        metadata["fold"] = _format_fold_name(fold)
    metadata.update(_safe_dict(extra))
    return metadata


def save_model_artifacts(
    *,
    backend: str,
    preset: str,
    model: Any,
    model_id: str | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    training_time: float | None = None,
    validation_score: float | None = None,
    metrics: dict[str, Any] | None = None,
    pipeline_steps: dict[str, Any] | None = None,
    fold: int | str | None = None,
    extra: dict[str, Any] | None = None,
    h2o_module: Any | None = None,
    artifact_extension: str = ".joblib",
) -> dict[str, Any]:
    """Persist only the final model artifact for an AutoML run."""
    artifact_dir = automl_models_dir(preset)
    resolved_model_id = _safe_artifact_name(
        model_id or model_name or f"{_safe_artifact_name(backend)}_model"
    )
    model_path = None
    save_error = None

    if model is None:
        save_error = "No fitted model was provided for persistence."
    elif h2o_module is not None:
        try:
            model_path = Path(h2o_module.save_model(model, path=str(artifact_dir), force=True))
        except Exception as exc:
            save_error = f"Failed to save H2O model: {exc}"
            logger.warning(save_error, exc_info=True)
    else:
        target_path = artifact_dir / f"{resolved_model_id}{artifact_extension}"
        try:
            model_path = _dump_sklearn_artifact(model, target_path)
        except Exception as exc:
            save_error = f"Failed to serialize model: {exc}"
            logger.warning(save_error, exc_info=True)

    return build_model_metadata(
        backend=backend,
        preset=str(preset or "default"),
        fold=fold,
        model_id=resolved_model_id,
        model_name=model_name or resolved_model_id,
        model_type=model_type,
        training_time=training_time,
        validation_score=validation_score,
        metrics=metrics,
        pipeline_steps=pipeline_steps,
        model_path=str(model_path) if model_path else None,
        extra={
            **_safe_dict(extra),
            "model_save_error": save_error,
        },
    )


def build_experiment_summary(
    report: dict[str, Any],
    framework_name: str | None = None,
) -> dict[str, Any]:
    saved_model = _safe_dict(report.get("saved_model"))
    history = [_compact_history_row(row) for row in report.get("model_history", [])]
    ranked_history = sorted(
        history,
        key=lambda row: (
            row.get("rank") is None,
            row.get("rank") if row.get("rank") is not None else 10**9,
        ),
    )
    winner_row = _winner_row(ranked_history, report.get("optimization_metric"))
    backend_artifacts = _safe_dict(report.get("backend_artifacts"))
    resolved_params = _safe_dict(backend_artifacts.get("resolved_params"))
    fold_metrics = _safe(backend_artifacts.get("fold_metrics"))

    return {
        "backend": report.get("backend") or framework_name,
        "framework_name": framework_name,
        "preset": report.get("preset"),
        "task_type": report.get("task_type"),
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "optimization_metric": report.get("optimization_metric"),
        "winner": {
            "model_id": saved_model.get("model_id") or winner_row.get("model_id"),
            "model_name": saved_model.get("model_name") or winner_row.get("model_name"),
            "model_type": saved_model.get("model_type") or winner_row.get("model_type"),
            "validation_score": saved_model.get("validation_score")
            if saved_model.get("validation_score") is not None
            else winner_row.get("validation_score"),
            "metrics": saved_model.get("metrics") or _safe_dict(report.get("final_metrics")),
            "hyperparameters": _winner_hyperparameters(saved_model, winner_row),
            "artifact_path": saved_model.get("model_path"),
        },
        "search_summary": _safe_dict(report.get("search_summary")),
        "metrics": _safe_dict(report.get("final_metrics")),
        "ranking": ranked_history,
        "fold_metrics": fold_metrics,
        "resolved_params": resolved_params,
        "saved_model": saved_model,
        "library_notes": _library_notes(report),
    }


def save_experiment_results(
    report: dict[str, Any],
    params: dict[str, Any] | None = None,
    framework_name: str | None = None,
) -> str:
    config = _resolve_params(params) if params else {"preset": report.get("preset")}
    preset_name = str(report.get("preset") or config.get("preset") or "default")
    summary = build_experiment_summary(report, framework_name=framework_name)
    report_path = automl_results_dir(preset_name) / EXPERIMENT_SUMMARY_FILENAME
    _write_json_artifact(report_path, summary)
    return str(report_path)


def _model_artifact_dir(
    backend: str | None = None,
    preset: str | None = None,
    fold: int | str | None = None,
) -> Path:
    del backend, fold
    return automl_models_dir(preset)


def _format_fold_name(fold: int | str) -> str:
    if isinstance(fold, int):
        return f"fold_{fold}"
    fold_value = str(fold)
    if fold_value.isdigit():
        return f"fold_{fold_value}"
    return fold_value


def _safe_artifact_name(value: Any) -> str:
    text = str(value or "model").strip()
    cleaned = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:160] or "model"


def _compact_history_row(row: dict[str, Any]) -> dict[str, Any]:
    backend_metadata = _safe_dict(row.get("backend_metadata"))
    return {
        "rank": _safe(row.get("rank")),
        "model_id": row.get("model_id") or row.get("model_name"),
        "model_name": row.get("model_name"),
        "model_type": row.get("model_type") or row.get("model_family"),
        "validation_score": _safe(row.get("validation_score")),
        "train_score": _safe(row.get("train_score")),
        "test_score": _safe(row.get("test_score")),
        "fit_time": _safe(row.get("fit_time")),
        "predict_time": _safe(row.get("predict_time")),
        "generation_or_iteration": _safe(row.get("generation_or_iteration")),
        "status": row.get("status"),
        "selected_in_final_ensemble": bool(row.get("selected_in_final_ensemble")),
        "hyperparameters": _extract_hyperparameters(row, backend_metadata),
        "backend_metadata": _compact_backend_metadata(backend_metadata),
    }


def _winner_row(
    ranking: list[dict[str, Any]],
    optimization_metric: str | None,
) -> dict[str, Any]:
    for row in ranking:
        if row.get("selected_in_final_ensemble"):
            return row
    scored_rows = [row for row in ranking if row.get("validation_score") is not None]
    if scored_rows:
        return sorted(
            scored_rows,
            key=lambda row: row["validation_score"],
            reverse=not _lower_is_better(optimization_metric),
        )[0]
    return ranking[0] if ranking else {}


def _winner_hyperparameters(
    saved_model: dict[str, Any],
    winner_row: dict[str, Any],
) -> dict[str, Any]:
    pipeline_steps = saved_model.get("pipeline_steps")
    if isinstance(pipeline_steps, dict):
        final_estimator = pipeline_steps.get("final_estimator")
        if isinstance(final_estimator, dict) and final_estimator.get("params"):
            return _safe_dict(final_estimator.get("params"))
    return _safe_dict(winner_row.get("hyperparameters"))


def _extract_hyperparameters(
    row: dict[str, Any],
    backend_metadata: dict[str, Any],
) -> dict[str, Any]:
    for key in ("hyperparameters", "params", "model_params"):
        value = row.get(key)
        if isinstance(value, dict):
            return _safe_dict(value)
    for key in ("params", "hyperparameters", "model_params"):
        value = backend_metadata.get(key)
        if isinstance(value, dict):
            return _safe_dict(value)
    return {}


def _compact_backend_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "model_id",
        "algo",
        "mean_test_score",
        "mean_train_score",
        "cost",
        "score",
        "rmse",
        "RMSE",
        "mae",
        "MAE",
        "mse",
        "MSE",
        "r2",
        "AUC",
        "auc",
        "logloss",
        "training_time_ms",
        "predict_time_per_row_ms",
        "eval_error",
    ]
    return {key: _safe(metadata[key]) for key in keep_keys if key in metadata}


def _library_notes(report: dict[str, Any]) -> list[str]:
    backend = str(report.get("backend") or "").lower()
    notes = []
    if "h2o" in backend:
        notes.append(
            "H2O may create temporary runtime files/logs while the Java cluster is active; "
            "the project persists only the final leader model and experiment_summary.json."
        )
        notes.append(
            "H2O leaderboard rows do not expose every candidate hyperparameter; load the saved "
            "leader model to inspect its full native parameters."
        )
    if "auto-sklearn" in backend:
        notes.append(
            "When auto-sklearn runs in Docker, the pickle/joblib artifact must be loaded in an "
            "environment with auto-sklearn installed."
        )
    return notes


def _write_json_artifact(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        logger.warning("Failed to write JSON artifact to %s", path, exc_info=True)


def _dump_sklearn_artifact(model: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(model, path)
        return path
    except Exception:
        fallback_path = path.with_suffix(".pkl")
        with open(fallback_path, "wb") as file_obj:
            pickle.dump(model, file_obj)
        if fallback_path != path:
            logger.info("joblib serialization failed; saved pickle artifact to %s", fallback_path)
        return fallback_path


def _model_name(model: Any) -> str | None:
    if model is None:
        return None
    return getattr(model, "model_id", None) or model.__class__.__name__
