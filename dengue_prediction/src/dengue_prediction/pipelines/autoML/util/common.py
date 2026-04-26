from __future__ import annotations

import copy
import inspect
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from dengue_prediction.settings import PROJECT_ROOT

def _resolve_params(params: dict[str, Any] | None) -> dict[str, Any]:
    raw = copy.deepcopy(params or {})
    presets = raw.pop("presets", {}) or {}
    preset = raw.pop("preset", "medium" if presets else None)
    overrides = raw.pop("overrides", {}) or {}
    task_type = raw.pop("task_type", "auto")
    optimization_metric = raw.pop("optimization_metric", "auto")

    if presets and preset not in presets:
        valid = ", ".join(sorted(presets))
        raise ValueError(f"Invalid preset '{preset}'. Expected one of: {valid}.")

    resolved = copy.deepcopy(presets.get(preset, {}))
    resolved.update({key: value for key, value in raw.items() if value is not None})
    resolved.update({key: value for key, value in overrides.items() if value is not None})

    return {
        "preset": preset,
        "task_type": task_type,
        "optimization_metric": optimization_metric,
        "params": resolved,
    }


def _infer_task_type(y: pd.Series, configured_task_type: str | None) -> str:
    if configured_task_type in {"classification", "regression"}:
        return configured_task_type
    if configured_task_type not in {None, "auto"}:
        raise ValueError("task_type must be 'auto', 'classification', or 'regression'.")

    target = pd.Series(y)
    target_kind = type_of_target(target.dropna())
    if target_kind in {"binary", "multiclass", "multilabel-indicator"}:
        unique_values = target.nunique(dropna=True)
        threshold = max(20, int(math.sqrt(max(len(target), 1)) * 2))
        if pd.api.types.is_numeric_dtype(target) and unique_values > threshold:
            return "regression"
        return "classification"
    return "regression"


def _columns_with_infinite_values(data: pd.DataFrame) -> list[str]:
    numeric_data = data.select_dtypes(include=[np.number])
    columns = []
    for column in numeric_data.columns:
        values = numeric_data[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            columns.append(str(column))
    return columns


def _series_has_infinite_values(data: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(data):
        return False
    values = data.to_numpy(dtype=float, na_value=np.nan)
    return bool(np.isinf(values).any())


def _ensure_docker_available_for_autosklearn() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError(
            "auto-sklearn is not supported natively on Windows. Install Docker Desktop "
            "to run `kedro run --pipeline autosklearn`."
        )

    docker_check = subprocess.run(
        ["docker", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if docker_check.returncode != 0:
        raise RuntimeError(
            "auto-sklearn on Windows requires a running Docker daemon. Start Docker Desktop "
            "or use the TPOT/H2O pipelines instead.\n"
            + _format_process_output(docker_check)
        )


def _autosklearn_dockerfile_path() -> Path:
    candidates = [
        PROJECT_ROOT / "Dockerfile.autosklearn",
        PROJECT_ROOT / "dengue_prediction" / "Dockerfile.autosklearn",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Dockerfile.autosklearn. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _format_process_output(result: subprocess.CompletedProcess) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    details = [f"exit_code: {result.returncode}"]
    if stdout:
        details.append(f"stdout:\n{stdout}")
    if stderr:
        details.append(f"stderr:\n{stderr}")
    if len(details) == 1:
        details.append("stdout/stderr: <empty>")
    return "\n".join(details)


def _read_autosklearn_docker_outputs(temp_path: Path) -> tuple[str, dict[str, Any]]:
    result_path = temp_path / "automl_result.json"
    reference_path = temp_path / "autosklearn_model_reference.json"
    missing = [path for path in [result_path, reference_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "auto-sklearn Docker finished but did not create expected output file(s): "
            + ", ".join(str(path) for path in missing)
        )

    with open(result_path, "r", encoding="utf-8") as file_obj:
        result = json.load(file_obj)
    with open(reference_path, "r", encoding="utf-8") as file_obj:
        model_reference_data = json.load(file_obj)

    artifact_relative_path = model_reference_data.get("model_artifact")
    if not artifact_relative_path:
        raise ValueError(
            f"{reference_path} is missing the 'model_artifact' field produced by Docker."
        )

    model_artifact_path = _resolve_project_artifact_path(artifact_relative_path)
    if not model_artifact_path.exists():
        raise FileNotFoundError(
            "auto-sklearn Docker reported a model artifact, but the host could not find it: "
            f"{model_artifact_path}"
        )

    artifacts = result.setdefault("backend_artifacts", {})
    artifacts["model_path"] = str(model_artifact_path)
    artifacts["model_reference"] = _safe_dict(model_reference_data)

    predictions_relative_path = artifacts.get("predictions_path")
    if predictions_relative_path:
        predictions_path = _resolve_project_artifact_path(predictions_relative_path)
        if not predictions_path.exists():
            raise FileNotFoundError(
                "auto-sklearn Docker reported a predictions artifact, but the host could not find it: "
                f"{predictions_path}"
            )
        predictions_df = pd.read_csv(predictions_path)
        artifacts["predictions_path"] = str(predictions_path)
        artifacts["predictions_row_count"] = int(len(predictions_df))
        artifacts["predictions_preview"] = _records(predictions_df.head(10))

    return str(model_artifact_path), result


def _resolve_project_artifact_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _get_signature_parameter_names(*callables: Any) -> set[str]:
    accepted_names: set[str] = set()
    for callable_obj in callables:
        signature = inspect.signature(callable_obj)
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            accepted_names.add(name)
    return accepted_names


def _pick_score(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        if column in row.index:
            value = _safe(row.get(column))
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _model_family(model: Any) -> str | None:
    if model is None:
        return None
    if hasattr(model, "steps") and getattr(model, "steps"):
        try:
            return model.steps[-1][1].__class__.__name__
        except Exception:
            return model.__class__.__name__
    return model.__class__.__name__


def _seconds_between(start_value: Any, end_value: Any) -> float | None:
    if start_value is None or end_value is None:
        return None
    try:
        start = pd.to_datetime(start_value)
        end = pd.to_datetime(end_value)
        if pd.isna(start) or pd.isna(end):
            return None
        return float((end - start).total_seconds())
    except Exception:
        return None


def _ms_to_seconds(value: Any) -> float | None:
    try:
        return float(value) / 1000.0 if value is not None else None
    except Exception:
        return None


def _call_if_exists(obj: Any, method_name: str):
    if not hasattr(obj, method_name):
        return None
    try:
        return getattr(obj, method_name)()
    except Exception:
        return None


def _get_attr_or_default(obj: Any, attr_name: str, default: Any = None):
    try:
        return getattr(obj, attr_name)
    except Exception:
        return default


def _lower_is_better(metric_name: str | None) -> bool:
    if not metric_name:
        return False
    metric_name = metric_name.lower()
    if metric_name.startswith("neg_"):
        return False
    return any(token in metric_name for token in ["loss", "error", "rmse", "mse", "mae", "deviance", "logloss"])


def _records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, pd.DataFrame) and not data.empty:
        return [_safe_dict(row) for row in data.to_dict(orient="records")]
    return []


def _safe_dict(data: dict[str, Any] | None) -> dict[str, Any]:
    return {key: _safe(value) for key, value in (data or {}).items()}


def _safe(value: Any):
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return _records(value)
    if isinstance(value, pd.Series):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return _safe_dict(value)
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "to_dict") and not isinstance(value, (str, bytes)):
        try:
            return _safe(value.to_dict())
        except Exception:
            return str(value)
    return value


def _cleanup_temp_files(paths: list[os.PathLike[str] | str]) -> None:
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            continue
