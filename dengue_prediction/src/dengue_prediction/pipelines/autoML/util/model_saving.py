from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_prediction.settings import DATA_DIR

from .common import _safe, _safe_dict

logger = logging.getLogger(__name__)

def build_model_metadata(
    backend: str,
    preset: str,
    fold: int | str,
    model_id: str | None,
    model_name: str | None,
    model_type: str | None,
    training_time: float | None,
    validation_score: float | None,
    metrics: dict[str, Any] | None,
    pipeline_steps: dict[str, Any] | None,
    model_path: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "backend": backend,
        "preset": preset,
        "fold": _format_fold_name(fold),
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
    metadata.update(_safe_dict(extra))
    return metadata


def _model_artifact_dir(backend: str, preset: str, fold: int | str) -> Path:
    fold_dir = (
        DATA_DIR
        / "results"
        / "models"
        / "autoML"
        / str(preset or "default")
        / backend
        / _format_fold_name(fold)
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def _format_fold_name(fold: int | str) -> str:
    if isinstance(fold, int):
        return f"fold_{fold}"
    fold_value = str(fold)
    if fold_value.isdigit():
        return f"fold_{fold_value}"
    return fold_value


def _timestamp_for_path() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%fZ")


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for counter in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{_timestamp_for_path()}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique artifact path for {path}")


def _write_json_artifact(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        logger.warning("Failed to write JSON artifact to %s", path, exc_info=True)


def _write_text_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        logger.warning("Failed to write text artifact to %s", path, exc_info=True)


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
