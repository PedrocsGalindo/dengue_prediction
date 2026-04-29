from __future__ import annotations

import copy
import json
import logging
import subprocess
import time
from typing import Any

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from dengue_prediction.settings import PROJECT_ROOT

from .util.common import (
    _infer_task_type,
    _autosklearn_dockerfile_path,
    _cleanup_temp_files,
    _ensure_docker_available_for_autosklearn,
    _format_process_output,
    _read_autosklearn_docker_outputs,
    _resolve_params,
)
from .util.h2o import run_h2o_automl
from .util.metrics import (
    _best_history_score,
    _predict_proba,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import math
from .util.model_saving import save_model_artifacts
from .util.reporting import (
    _autosklearn_history,
    build_autosklearn_report,
    save_automl_report,
)
from .util.tpot import run_tpot_automl

logger = logging.getLogger(__name__)


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
    return run_tpot_automl(X, y, params, n_splits)


def autoML_h2o(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    n_splits: int = 5,
):
    return run_h2o_automl(X, y, params, n_splits)


def autoML_sklearn_docker(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    n_splits: int = 5,
):
    _ensure_docker_available_for_autosklearn()

    image_name = "autosklearn"
    dockerfile = _autosklearn_dockerfile_path()
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
            raise RuntimeError(
                "Failed to build the auto-sklearn Docker image.\n"
                + _format_process_output(build_result)
            )

    temps_path = PROJECT_ROOT / "temp"
    temps_path.mkdir(exist_ok=True)

    output_paths = [
        temps_path / "automl_result.json",
        temps_path / "autosklearn_model_reference.json",
    ]
    _cleanup_temp_files(output_paths)

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
            f"{PROJECT_ROOT.as_posix()}:/workspace",
            image_name,
            "/workspace/temp",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run_result.returncode != 0:
        raise RuntimeError(
            "Failed to run auto-sklearn inside Docker.\n"
            + _format_process_output(run_result)
        )

    model_reference, result = _read_autosklearn_docker_outputs(temps_path)

    _cleanup_temp_files(
        [
            temps_path / "temp_autosklearn_params.json",
            temps_path / "temp_autosklearn_X.csv",
            temps_path / "temp_autosklearn_Y.csv",
            temps_path / "temp_autosklearn_n_splits.txt",
            temps_path / "automl_result.json",
            temps_path / "autosklearn_model_reference.json",
        ]
    )
    try:
        temps_path.rmdir()
    except OSError:
        pass
    return model_reference, result


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
    X = X.drop(columns=["data"], errors="ignore")

    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = estimator_class(**automl_params)
        fold_model.fit(X_train.copy(), y_train.copy())
        fold_pred = fold_model.predict(X_test)
        fold_proba = _predict_proba(fold_model, X_test)
        fold_mse = mean_squared_error(y_test, fold_pred) 
        fold_metrics.append({
            "mae": float(mean_absolute_error(y_test, fold_pred)),
            "mse": float(fold_mse),
            "rmse": float(math.sqrt(fold_mse)),
            "r2": float(r2_score(y_test, fold_pred)),
        })

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

    result = build_autosklearn_report(
        config=config,
        task_type=task_type,
        automl=automl,
        fold_metrics=fold_metrics,
        optimization_metric=optimization_metric,
        history=history,
        search_time=search_time,
        notes=notes,
        automl_params=automl_params,
        leaderboard_df=leaderboard_df,
    )
    model_metadata = save_model_artifacts(
        backend="auto-sklearn",
        preset=str(config["preset"] or "default"),
        model=automl,
        model_id="autosklearn_model",
        model_name=automl.__class__.__name__,
        model_type=automl.__class__.__name__,
        training_time=search_time,
        validation_score=_best_history_score(history, optimization_metric),
        metrics=result.get("final_metrics"),
        extra={
            "load_in": "Python environment with auto-sklearn installed",
        },
    )
    result["saved_model"] = model_metadata
    result.setdefault("backend_artifacts", {})["model_path"] = model_metadata.get("model_path")
    return automl, result
