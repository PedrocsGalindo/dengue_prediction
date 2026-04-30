from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dengue_prediction.settings import DATA_DIR, PROJECT_ROOT

from .reports import save_automl_outputs


def run_h2o_automl(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None,
):
    import h2o
    from h2o.automl import H2OAutoML, get_leaderboard

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = DATA_DIR / "results" / "autoML" / "h2o" / run_id
    export_checkpoints_dir = DATA_DIR / "results" / "autoML" / "h2o" / run_id / "checkpoints"
    h2o_params, metric, preset, use_leaderboard_frame = _prepare_params(params)
    print(f"preset -> {preset}")
    h2o.init()
    target_name = y.name
    project_name = f"dengue_h2o_{run_id.replace('-', '').replace(':', '')}"

    try:
        x_cols = [x for x in X.columns]
        df = h2o.H2OFrame(X.join(y))
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
            export_checkpoints_dir=str(export_checkpoints_dir),
            project_name=project_name,
        )
        aml.train(x=x_cols, y=target_name, training_frame=df)
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
        print(f"\n\n Leader Board:")
        print(leaderboard_df)
        print(f"\n\n Event log:")
        print(aml.event_log)
        print(f"\n\n training_info:")
        print(aml.training_info)    
        result = {
            "search_history": leaderboard_df,
            "best_model": {
                "model_name": leader_id,
                "model_family": leader_row.get("algo") or _safe(getattr(aml.leader, "algo", None)),
                "hyperparameters": _safe(getattr(aml.leader, "params", {})), #  nao ta retornando nada 
                "score": {
                    "metric": metric,
                    "value": _safe(leader_row.get(leader_score_column)),
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
                "use_leaderboard_frame": use_leaderboard_frame,
                "training_time_seconds": float(training_time),
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
    use_leaderboard_frame = bool(params.pop("use_leaderboard_frame", True))

    if preset not in presets:
        raise ValueError(f"Unknown H2O preset: {preset}")

    resolved = presets[preset]
    return resolved, metric, preset, use_leaderboard_frame


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
