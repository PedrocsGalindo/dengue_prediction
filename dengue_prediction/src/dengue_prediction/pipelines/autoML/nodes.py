from __future__ import annotations

import copy
import inspect
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.multiclass import type_of_target

from dengue_prediction.settings import DATA_DIR, PROJECT_ROOT


def get_dataset(recife_dengue_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = recife_dengue_data.drop(columns=["casos_dengue"])
    y = recife_dengue_data["casos_dengue"]
    return X, y


def autoML_tpot(
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

    fold_metrics = []
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = _make_tpot_estimator(estimator_class, tpot_params)
        _fit_tpot_estimator(fold_model, X_train, y_train)
        fold_pred = fold_model.predict(X_test)
        fold_proba = _predict_proba(fold_model, X_test)
        fold_metrics.append(_compute_metrics(task_type, y_test, fold_pred, fold_proba))

    started_at = time.perf_counter()
    final_search = _make_tpot_estimator(estimator_class, tpot_params)
    _fit_tpot_estimator(final_search, X, y)
    search_time = time.perf_counter() - started_at

    history = _tpot_history(final_search, optimization_metric)
    result = {
        "backend": "TPOT",
        "preset": config["preset"],
        "task_type": task_type,
        "final_model_repr": str(final_search.fitted_pipeline_),
        "final_model_name": str(final_search.fitted_pipeline_),
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
    return final_search.fitted_pipeline_, result


def autoML_h2o(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    n_splits: int = 5,
):
    config = _resolve_params(params)
    task_type = _infer_task_type(y, config["task_type"])
    optimization_metric = config["optimization_metric"]
    notes = [f"preset={config['preset']}"] if config["preset"] else []

    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError as exc:
        raise ImportError(f"Failed to import H2O AutoML: {exc}") from exc

    h2o_params, h2o_init_params, use_leaderboard_frame = _prepare_h2o_params(
        config["params"], H2OAutoML
    )
    include_algos = h2o_params.get("include_algos")
    exclude_algos = h2o_params.get("exclude_algos")
    if include_algos and exclude_algos:
        raise ValueError("H2O AutoML does not allow include_algos and exclude_algos together.")

    h2o_params["project_name"] = h2o_params.get("project_name") or f"dengue_automl_{uuid.uuid4().hex[:8]}"
    _ensure_h2o_connection(h2o, h2o_init_params)

    target_name = y.name or "target"
    fold_metrics = []
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_df = X_train.copy()
        train_df[target_name] = y_train.values
        test_df = X_test.copy()
        test_df[target_name] = y_test.values

        train_h2o = h2o.H2OFrame(train_df)
        test_h2o = h2o.H2OFrame(test_df)
        if task_type == "classification":
            train_h2o[target_name] = train_h2o[target_name].asfactor()
            test_h2o[target_name] = test_h2o[target_name].asfactor()

        x_cols = [column for column in train_h2o.columns if column != target_name]
        fold_params = copy.deepcopy(h2o_params)
        fold_params["project_name"] = f"{h2o_params['project_name']}_fold_{len(fold_metrics) + 1}"
        fold_aml = H2OAutoML(**fold_params)
        fold_aml.train(
            x=x_cols,
            y=target_name,
            training_frame=train_h2o,
            leaderboard_frame=test_h2o if use_leaderboard_frame else None,
        )
        fold_metrics.append(_h2o_metrics(fold_aml.leader, test_h2o, target_name, task_type))

    full_df = X.copy()
    full_df[target_name] = y.values
    full_h2o = h2o.H2OFrame(full_df)
    if task_type == "classification":
        full_h2o[target_name] = full_h2o[target_name].asfactor()
    x_cols = [column for column in full_h2o.columns if column != target_name]

    started_at = time.perf_counter()
    aml = H2OAutoML(**h2o_params)
    aml.train(x=x_cols, y=target_name, training_frame=full_h2o)
    search_time = time.perf_counter() - started_at

    leaderboard_df = _h2o_to_pandas(_get_h2o_leaderboard(aml))
    event_log_df = _h2o_to_pandas(getattr(aml, "event_log", None))
    history = _h2o_history(aml, optimization_metric)

    result = {
        "backend": "H2O AutoML",
        "preset": config["preset"],
        "task_type": task_type,
        "final_model_repr": getattr(aml.leader, "model_id", "unknown"),
        "final_model_name": getattr(aml.leader, "model_id", "unknown"),
        "final_metrics": _aggregate_metrics(fold_metrics),
        "optimization_metric": optimization_metric,
        "model_history": history,
        "search_summary": _search_summary(history, search_time, optimization_metric, notes),
        "backend_artifacts": {
            "resolved_params": _safe_dict(h2o_params),
            "leaderboard": _records(leaderboard_df),
            "event_log": _records(event_log_df),
            "training_info": _safe_dict(getattr(aml, "training_info", {})),
            "fold_metrics": _safe(fold_metrics),
            "init_params": _safe_dict(h2o_init_params),
        },
    }
    return aml.leader, result


def autoML_sklearn_docker(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    n_splits: int = 5,
):
    _ensure_docker_available_for_autosklearn()

    image_name = "autosklearn"
    dockerfile = PROJECT_ROOT / "Dockerfile.autosklearn"
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect_result.returncode != 0:
        build_result = subprocess.run(
            ["docker", "build", "-f", str(dockerfile), "-t", image_name, str(PROJECT_ROOT)],
            capture_output=True,
            text=True,
            check=False,
        )
        if build_result.returncode != 0:
            stderr = (build_result.stderr or build_result.stdout or "").strip()
            raise RuntimeError(f"Failed to build the auto-sklearn Docker image: {stderr}")

    temps_path = PROJECT_ROOT / "temp"
    temps_path.mkdir(exist_ok=True)

    params_path = temps_path / "temp_autosklearn_params.json"
    with open(params_path, "w", encoding="utf-8") as file_obj:
        json.dump(params, file_obj)
    X.to_csv(temps_path / "temp_autosklearn_X.csv", index=False)
    y.to_csv(temps_path / "temp_autosklearn_Y.csv", index=False)
    with open(temps_path / "temp_autosklearn_n_splits.txt", "w", encoding="utf-8") as file_obj:
        file_obj.write(str(n_splits))

    run_result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{PROJECT_ROOT}:/workspace",
            image_name,
            "/workspace/temp",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run_result.returncode != 0:
        stderr = (run_result.stderr or run_result.stdout or "").strip()
        raise RuntimeError(f"Failed to run auto-sklearn inside Docker: {stderr}")

    automl = joblib.load(temps_path / "automl.pkl")
    with open(temps_path / "automl_result.json", "r", encoding="utf-8") as file_obj:
        result = json.load(file_obj)

    _cleanup_temp_files(
        [
            temps_path / "temp_autosklearn_params.json",
            temps_path / "temp_autosklearn_X.csv",
            temps_path / "temp_autosklearn_Y.csv",
            temps_path / "temp_autosklearn_n_splits.txt",
            temps_path / "automl.pkl",
            temps_path / "automl_result.json",
        ]
    )
    try:
        temps_path.rmdir()
    except OSError:
        pass
    return automl, result


def autoML_sklearn(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    n_splits: int = 5,
):
    config = _resolve_params(params)
    task_type = _infer_task_type(y, config["task_type"])
    optimization_metric = config["optimization_metric"]
    automl_params = copy.deepcopy(config["params"])
    notes = [f"preset={config['preset']}"] if config["preset"] else []

    try:
        if task_type == "classification":
            import autosklearn.classification as autosklearn_module

            estimator_class = autosklearn_module.AutoSklearnClassifier
        else:
            import autosklearn.regression as autosklearn_module

            estimator_class = autosklearn_module.AutoSklearnRegressor
    except ImportError as exc:
        raise ImportError(f"Failed to import auto-sklearn: {exc}") from exc

    fold_metrics = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    X = X.drop(columns=["data"], errors="ignore")

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = estimator_class(**automl_params)
        fold_model.fit(X_train.copy(), y_train.copy())
        fold_pred = fold_model.predict(X_test)
        fold_proba = _predict_proba(fold_model, X_test)
        fold_metrics.append(_compute_metrics(task_type, y_test, fold_pred, fold_proba))

    started_at = time.perf_counter()
    automl = estimator_class(**automl_params)
    automl.fit(X.copy(), y.copy())
    search_time = time.perf_counter() - started_at

    history = _autosklearn_history(automl, optimization_metric)
    leaderboard_df = pd.DataFrame()
    if hasattr(automl, "leaderboard"):
        try:
            leaderboard_df = automl.leaderboard(detailed=True)
        except TypeError:
            try:
                leaderboard_df = automl.leaderboard()
            except Exception:
                leaderboard_df = pd.DataFrame()
        except Exception:
            leaderboard_df = pd.DataFrame()

    result = {
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
            "performance_over_time": _records(getattr(automl, "performance_over_time_", pd.DataFrame())),
            "sprint_statistics": _safe(_call_if_exists(automl, "sprint_statistics")),
            "cv_results": _safe_dict(getattr(automl, "cv_results_", {})),
            "fold_metrics": _safe(fold_metrics),
        },
    }
    return automl, result


def save_automl_report(
    report: dict[str, Any],
    params: dict[str, Any],
    framework_name: str,
) -> str:
    config = _resolve_params(params)
    preset_name = str(config["preset"] or "default")
    report_dir = DATA_DIR / "results" / "autoML" / preset_name
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"{framework_name}_report.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2, default=str)

    return str(report_path)


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


def _ensure_docker_available_for_autosklearn() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError(
            "auto-sklearn is not supported natively on Windows. Install Docker Desktop "
            "to run `kedro run --pipeline autosklearn`."
        )

    docker_check = subprocess.run(
        ["docker", "version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if docker_check.returncode != 0:
        raise RuntimeError(
            "auto-sklearn on Windows requires a running Docker daemon. Start Docker Desktop "
            "or use the TPOT/H2O pipelines instead."
        )


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


def _compute_metrics(
    task_type: str,
    y_true: pd.Series,
    y_pred: Any,
    y_proba: Any = None,
) -> dict[str, Any]:
    if task_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc": None,
        }
        try:
            if y_proba is not None:
                proba = np.asarray(y_proba)
                if y_true.nunique(dropna=True) == 2:
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        metrics["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
                    else:
                        metrics["roc_auc"] = roc_auc_score(y_true, proba)
                elif proba.ndim == 2:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true,
                        proba,
                        multi_class="ovr",
                        average="weighted",
                    )
        except Exception:
            metrics["roc_auc"] = None
        return _safe_dict(metrics)

    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


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

    cv_results = pd.DataFrame(getattr(estimator, "cv_results_", {}))
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


def _h2o_metrics(model, test_frame, target_name: str, task_type: str) -> dict[str, Any]:
    y_true = _h2o_to_pandas(test_frame[target_name]).iloc[:, 0]
    predictions = _h2o_to_pandas(model.predict(test_frame))
    if task_type == "classification":
        y_pred = predictions.iloc[:, 0]
        y_proba = predictions.iloc[:, 1:] if predictions.shape[1] > 1 else None
        return _compute_metrics(task_type, y_true, y_pred, y_proba)
    return _compute_metrics(task_type, y_true, predictions.iloc[:, 0])


def _predict_proba(model, X):
    if X is None or not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(X)
    except Exception:
        return None


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


def _h2o_to_pandas(frame: Any) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    try:
        return frame.as_data_frame()
    except Exception:
        return pd.DataFrame()


def _call_if_exists(obj: Any, method_name: str):
    if not hasattr(obj, method_name):
        return None
    try:
        return getattr(obj, method_name)()
    except Exception:
        return None


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
