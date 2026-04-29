from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from dengue_prediction.settings import DATA_DIR, PROJECT_ROOT

from .reports import save_automl_outputs


def run_h2o_automl(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None,
    n_splits: int = 5,
):
    import h2o
    from h2o.automl import H2OAutoML, get_leaderboard

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = DATA_DIR / "results" / "autoML" / "h2o" / run_id
    h2o_params, init_params, metric, preset, use_leaderboard_frame = _prepare_params(params)
    h2o_runtime_dir = PROJECT_ROOT / "temp" / "h2o_runtime"
    h2o_runtime_dir.mkdir(parents=True, exist_ok=True)
    h2o.init(
        max_mem_size=init_params.get("max_mem_size"),
        nthreads=init_params.get("nthreads"),
        ice_root=str(h2o_runtime_dir),
        log_dir=str(h2o_runtime_dir),
        verbose=False,
    )

    target_name = "regressao"
    project_name = f"dengue_h2o_{run_id.replace('-', '').replace(':', '')}"
    fold_metrics = []

    full_df = X.copy()
    full_df[target_name] = y 

    try:
        if n_splits >= 2:
            for fold_number, (train_idx, test_idx) in enumerate(
                TimeSeriesSplit(n_splits=n_splits).split(full_df),
                start=1,
            ):
                fold_train = full_df.iloc[train_idx].copy()
                fold_test = full_df.iloc[test_idx].copy()
                

                train_frame = h2o.H2OFrame(fold_train)
                test_frame = h2o.H2OFrame(fold_test)
                x_cols = [column for column in train_frame.columns if column != target_name]

                fold_aml = H2OAutoML(
                    max_runtime_secs=h2o_params.get("max_runtime_secs"),
                    max_models=h2o_params.get("max_models"),
                    max_runtime_secs_per_model=h2o_params.get("max_runtime_secs_per_model"),
                    nfolds=h2o_params.get("nfolds"),
                    stopping_metric=h2o_params.get("stopping_metric"),
                    sort_metric=h2o_params.get("sort_metric"),
                    stopping_rounds=h2o_params.get("stopping_rounds"),
                    stopping_tolerance=h2o_params.get("stopping_tolerance"),
                    seed=h2o_params.get("seed"),
                    verbosity=h2o_params.get("verbosity"),
                    project_name=f"{project_name}_fold_{fold_number}",
                )
                fold_aml.train(
                    x=x_cols,
                    y=target_name,
                    training_frame=train_frame,
                    leaderboard_frame=test_frame if use_leaderboard_frame else None,
                )

                predictions = _h2o_to_pandas(fold_aml.leader.predict(test_frame))
                fold_metrics.append(
                    _regression_metrics(fold_test[target_name], predictions.iloc[:, 0])
                )
        # treina com tudo
        x_cols = [column for column in full_df.columns if column != target_name]
        full_frame = h2o.H2OFrame(full_df)
        started_at = time.perf_counter()
        aml = H2OAutoML(
            max_runtime_secs=h2o_params.get("max_runtime_secs"),
            max_models=h2o_params.get("max_models"),
            max_runtime_secs_per_model=h2o_params.get("max_runtime_secs_per_model"),
            nfolds=h2o_params.get("nfolds"),
            stopping_metric=h2o_params.get("stopping_metric"),
            sort_metric=h2o_params.get("sort_metric"),
            stopping_rounds=h2o_params.get("stopping_rounds"),
            stopping_tolerance=h2o_params.get("stopping_tolerance"),
            seed=h2o_params.get("seed"),
            verbosity=h2o_params.get("verbosity"),
            project_name=project_name,
        )
        aml.train(x=x_cols, y=target_name, training_frame=full_frame)
        training_time = time.perf_counter() - started_at

        if aml.leader is None:
            raise RuntimeError("H2O AutoML completed without a leader model.")

        artifact_dir = output_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(h2o.save_model(aml.leader, path=str(artifact_dir), force=True))
        leaderboard_df = _h2o_to_pandas(get_leaderboard(aml, extra_columns="ALL"))

        leader_id = aml.leader.model_id
        leader_row = pd.Series(dtype=object)
        if not leaderboard_df.empty and "model_id" in leaderboard_df.columns and leader_id:
            matches = leaderboard_df[leaderboard_df["model_id"] == leader_id]
            if not matches.empty:
                leader_row = matches.iloc[0]
        leader_score_column = _score_column(leader_row, metric)

        mean_metrics = {}
        if fold_metrics:
            mean_metrics = {
                metric_name: float(np.mean([fold[metric_name] for fold in fold_metrics]))
                for metric_name in fold_metrics[0]
            }

        result = {
            "search_history": _search_history(leaderboard_df, metric, leader_id),
            "best_model": {
                "model_name": leader_id,
                "model_family": leader_row.get("algo") or _safe(getattr(aml.leader, "algo", None)),
                "hyperparameters": _safe(getattr(aml.leader, "params", {})),
                "score": {
                    "metric": metric,
                    "value": _safe(leader_row.get(leader_score_column))
                    if leader_score_column
                    else None,
                    "source": "leaderboard",
                },
                "artifact_path": model_path,
                "artifacts": {"artifact_path": model_path},
                "extra": {
                    "raw_leaderboard_row": _safe(leader_row.to_dict()),
                    "training_info": _safe(getattr(aml, "training_info", {})),
                },
            },
            "metadata": {
                "backend": "H2O AutoML",
                "task_type": "regression",
                "run_id": run_id,
                "preset": preset,
                "optimization_metric": metric,
                "resolved_params": _safe(h2o_params),
                "init_params": _safe(init_params),
                "use_leaderboard_frame": use_leaderboard_frame,
                "row_count": int(len(full_df)),
                "feature_count": int(full_df.shape[1]),
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
        return model_path, result, leaderboard_df
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
        except Exception:
            pass

def _prepare_params(
    params: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], str, str | None, bool]:
    presets = params.pop("presets", {}) or {}
    preset = params.pop("preset", None)
    params.pop("task_type", None)
    metric = str(params.pop("optimization_metric", "RMSE"))
    init_params = params.pop("init", {}) or {}
    use_leaderboard_frame = bool(params.pop("use_leaderboard_frame", True))

    if preset not in presets:
        raise ValueError(f"Unknown H2O preset: {preset}")

    resolved = presets[preset]
    return resolved, init_params, metric, preset, use_leaderboard_frame


def _regression_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _search_history(
    leaderboard_df: pd.DataFrame,
    metric: str,
    leader_id: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for iteration, (_, row) in enumerate(leaderboard_df.iterrows(), start=1):
        score_column = _score_column(row, metric)
        rows.append(
            {
                "iteration": iteration,
                "rank": iteration,
                "model_name": row.get("model_id"),
                "model_family": row.get("algo"),
                "hyperparameters": {},
                "score": {
                    "metric": metric,
                    "value": _safe(row.get(score_column)) if score_column else None,
                    "source_column": score_column,
                },
                "status": "ok",
                "selected_as_best": row.get("model_id") == leader_id,
                "extra": {"raw_leaderboard_row": _safe(row.to_dict())},
            }
        )
    return rows


def _score_column(row: pd.Series, metric: str) -> str | None:
    candidates = [
        metric,
        metric.lower(),
        metric.upper(),
        "rmse",
        "RMSE",
        "mae",
        "MAE",
        "mse",
        "MSE",
        "r2",
    ]
    return next((column for column in candidates if column in row.index), None)


def _h2o_to_pandas(frame: Any) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    return frame.as_data_frame()


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
