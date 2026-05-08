from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dengue_prediction.settings import DATA_DIR


def train_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Train a chronological linear regression baseline for dengue cases."""
    config = _resolve_params(params)
    X_model = _prepare_features(X)
    y_model = pd.Series(y).reset_index(drop=True)

    X_train, X_test, y_train, y_test = _chronological_train_test_split(
        X_model,
        y_model,
        test_size=config["test_size"],
    )

    fold_metrics = _cross_validate(
        X_train,
        y_train,
        n_splits=config["n_splits"],
        fit_intercept=config["fit_intercept"],
        positive=config["positive"],
        scale_features=config["scale_features"],
    )

    model = _build_model(
        fit_intercept=config["fit_intercept"],
        positive=config["positive"],
        scale_features=config["scale_features"],
    )
    started_at = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - started_at

    y_pred = model.predict(X_test)
    test_metrics = _regression_metrics(y_test, y_pred)
    predictions = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": y_pred,
            "residual": y_test.reset_index(drop=True) - y_pred,
        }
    )

    report = _build_report(
        model=model,
        config=config,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        fold_metrics=fold_metrics,
        test_metrics=test_metrics,
        training_time=training_time,
    )
    _save_artifacts(model, report, predictions)

    return model, report, predictions


def _resolve_params(params: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(params or {})
    return {
        "test_size": float(params.get("test_size", 0.2)),
        "n_splits": int(params.get("n_splits", 5)),
        "fit_intercept": bool(params.get("fit_intercept", True)),
        "positive": bool(params.get("positive", False)),
        "scale_features": bool(params.get("scale_features", True)),
    }


def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    features = X.copy().reset_index(drop=True)
    datetime_columns = list(features.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)

    for column in datetime_columns:
        values = pd.to_datetime(features[column], errors="coerce")
        features[f"{column}_ano"] = values.dt.year
        features[f"{column}_mes"] = values.dt.month
        features[f"{column}_dia"] = values.dt.day
        features[f"{column}_dia_da_semana"] = values.dt.dayofweek

    return features.drop(columns=datetime_columns, errors="ignore")


def _chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    if len(X) < 3:
        raise ValueError("At least 3 rows are required for a train/test split.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(X) * (1 - test_size))
    split_index = min(max(split_index, 1), len(X) - 1)

    return (
        X.iloc[:split_index].reset_index(drop=True),
        X.iloc[split_index:].reset_index(drop=True),
        y.iloc[:split_index].reset_index(drop=True),
        y.iloc[split_index:].reset_index(drop=True),
    )


def _cross_validate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    fit_intercept: bool,
    positive: bool,
    scale_features: bool,
) -> list[dict[str, float]]:
    max_splits = max(0, len(X_train) - 1)
    if n_splits < 2 or max_splits < 2:
        return []

    fold_metrics: list[dict[str, float]] = []
    for fold, (train_idx, valid_idx) in enumerate(
        TimeSeriesSplit(n_splits=min(n_splits, max_splits)).split(X_train),
        start=1,
    ):
        fold_model = _build_model(
            fit_intercept=fit_intercept,
            positive=positive,
            scale_features=scale_features,
        )
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_pred = fold_model.predict(X_train.iloc[valid_idx])
        metrics = _regression_metrics(y_train.iloc[valid_idx], fold_pred)
        metrics["fold"] = float(fold)
        fold_metrics.append(metrics)

    return fold_metrics


def _build_model(
    fit_intercept: bool,
    positive: bool,
    scale_features: bool,
) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(numeric_steps),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                make_column_selector(dtype_include=object),
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LinearRegression(
                    fit_intercept=fit_intercept,
                    positive=positive,
                ),
            ),
        ]
    )


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_report(
    model: Pipeline,
    config: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fold_metrics: list[dict[str, float]],
    test_metrics: dict[str, float],
    training_time: float,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "metadata": {
            "backend": "scikit-learn",
            "model_name": "LinearRegression",
            "task_type": "regression",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_time_seconds": float(training_time),
        },
        "methodology": {
            "split_strategy": "chronological holdout",
            "validation_strategy": "TimeSeriesSplit on training data",
            "metrics": ["MAE", "MSE", "RMSE", "R2"],
            "params": config,
        },
        "data": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_count": int(X_train.shape[1]),
            "target_name": str(y_train.name or "target"),
            "target_train_mean": float(y_train.mean()),
            "target_test_mean": float(y_test.mean()),
        },
        "cross_validation": {
            "fold_metrics": fold_metrics,
            "mean_metrics": _mean_fold_metrics(fold_metrics),
        },
        "test_metrics": test_metrics,
        "coefficients": _coefficients(model),
    }


def _mean_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not fold_metrics:
        return {}
    metric_names = [name for name in fold_metrics[0] if name != "fold"]
    return {
        name: float(np.mean([fold[name] for fold in fold_metrics]))
        for name in metric_names
    }


def _coefficients(model: Pipeline) -> list[dict[str, Any]]:
    linear_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{index}" for index in range(len(linear_model.coef_))]

    coefficients = [
        {
            "feature": str(feature_name),
            "coefficient": float(coefficient),
            "abs_coefficient": float(abs(coefficient)),
        }
        for feature_name, coefficient in zip(feature_names, np.ravel(linear_model.coef_))
    ]
    return sorted(coefficients, key=lambda item: item["abs_coefficient"], reverse=True)


def _save_artifacts(
    model: Pipeline,
    report: dict[str, Any],
    predictions: pd.DataFrame,
) -> dict[str, str]:
    run_id = report["metadata"]["run_id"]
    output_dir = DATA_DIR / "results" / "linear_regression" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "linear_regression.joblib"
    report_path = output_dir / "report.json"
    predictions_path = output_dir / "predictions.csv"

    artifacts = {
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "report_path": str(report_path),
        "predictions_path": str(predictions_path),
    }
    report["artifacts"] = artifacts

    joblib.dump(model, model_path)
    predictions.to_csv(predictions_path, index=False)
    with open(report_path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2, default=str)

    return artifacts
