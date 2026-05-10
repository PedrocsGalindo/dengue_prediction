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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dengue_prediction.settings import DATA_DIR

DATE_COLUMN = "data"


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
    X_train_features = _model_features(X_train)
    X_test_features = _model_features(X_test)

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
    model.fit(X_train_features, y_train)
    training_time = time.perf_counter() - started_at

    y_pred = model.predict(X_test_features)
    test_metrics = _regression_metrics(y_test, y_pred)
    predictions = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": y_pred,
            "residual": y_test.reset_index(drop=True) - y_pred,
        }
    )
    if DATE_COLUMN in X_test.columns:
        predictions.insert(0, DATE_COLUMN, X_test[DATE_COLUMN].reset_index(drop=True))

    report = _build_report(
        model=model,
        config=config,
        X_train=X_train,
        X_test=X_test,
        X_train_features=X_train_features,
        X_test_features=X_test_features,
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
    if DATE_COLUMN in features.columns:
        features[DATE_COLUMN] = pd.to_datetime(features[DATE_COLUMN], errors="coerce")
    elif {"ano", "mes", "dia"}.issubset(features.columns):
        features[DATE_COLUMN] = pd.to_datetime(
            features[["ano", "mes", "dia"]].rename(
                columns={"ano": "year", "mes": "month", "dia": "day"}
            ),
            errors="coerce",
        )

    return features


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

    data = X.copy()
    data["_target"] = y.to_numpy()
    if DATE_COLUMN in data.columns:
        data = data.dropna(subset=[DATE_COLUMN])
        data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
        unique_dates = pd.Series(data[DATE_COLUMN].unique()).sort_values().reset_index(drop=True)
        if len(unique_dates) < 3:
            raise ValueError("At least 3 unique dates are required for a chronological split.")

        split_date_index = int(len(unique_dates) * (1 - test_size))
        split_date_index = min(max(split_date_index, 1), len(unique_dates) - 1)
        cutoff_date = unique_dates.iloc[split_date_index]
        train_mask = data[DATE_COLUMN] < cutoff_date
        test_mask = data[DATE_COLUMN] >= cutoff_date

        train = data.loc[train_mask].reset_index(drop=True)
        test = data.loc[test_mask].reset_index(drop=True)
        return (
            train.drop(columns=["_target"]),
            test.drop(columns=["_target"]),
            train["_target"].rename(y.name),
            test["_target"].rename(y.name),
        )

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
    if n_splits < 2:
        return []

    fold_indices = _time_series_fold_indices(X_train, n_splits)
    fold_metrics: list[dict[str, float]] = []
    for fold, (train_idx, valid_idx) in enumerate(fold_indices, start=1):
        fold_model = _build_model(
            fit_intercept=fit_intercept,
            positive=positive,
            scale_features=scale_features,
        )
        fold_model.fit(_model_features(X_train.iloc[train_idx]), y_train.iloc[train_idx])
        fold_pred = fold_model.predict(_model_features(X_train.iloc[valid_idx]))
        metrics = _regression_metrics(y_train.iloc[valid_idx], fold_pred)
        metrics["fold"] = float(fold)
        fold_metrics.append(metrics)

    return fold_metrics


def _time_series_fold_indices(
    X_train: pd.DataFrame,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if DATE_COLUMN not in X_train.columns:
        max_splits = max(0, len(X_train) - 1)
        if max_splits < 2:
            return []
        from sklearn.model_selection import TimeSeriesSplit

        return list(TimeSeriesSplit(n_splits=min(n_splits, max_splits)).split(X_train))

    unique_dates = pd.Series(X_train[DATE_COLUMN].dropna().unique()).sort_values().reset_index(drop=True)
    if len(unique_dates) < 3:
        return []

    max_splits = len(unique_dates) - 1
    if max_splits < 2:
        return []

    from sklearn.model_selection import TimeSeriesSplit

    date_splits = TimeSeriesSplit(n_splits=min(n_splits, max_splits)).split(unique_dates)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    dates = X_train[DATE_COLUMN]
    for train_date_idx, valid_date_idx in date_splits:
        train_dates = set(unique_dates.iloc[train_date_idx])
        valid_dates = set(unique_dates.iloc[valid_date_idx])
        train_idx = np.flatnonzero(dates.isin(train_dates).to_numpy())
        valid_idx = np.flatnonzero(dates.isin(valid_dates).to_numpy())
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds


def _model_features(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=[DATE_COLUMN], errors="ignore")


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
    X_train_features: pd.DataFrame,
    X_test_features: pd.DataFrame,
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
            "split_strategy": "chronological holdout by date",
            "validation_strategy": "TimeSeriesSplit by date on training data",
            "metrics": ["MAE", "MSE", "RMSE", "R2"],
            "params": config,
        },
        "data": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_count": int(X_train_features.shape[1]),
            "target_name": str(y_train.name or "target"),
            "target_train_mean": float(y_train.mean()),
            "target_test_mean": float(y_test.mean()),
            "train_start_date": _date_boundary(X_train, "min"),
            "train_end_date": _date_boundary(X_train, "max"),
            "test_start_date": _date_boundary(X_test, "min"),
            "test_end_date": _date_boundary(X_test, "max"),
            "dropped_non_model_columns": [DATE_COLUMN] if DATE_COLUMN in X_train.columns else [],
        },
        "cross_validation": {
            "fold_metrics": fold_metrics,
            "mean_metrics": _mean_fold_metrics(fold_metrics),
        },
        "test_metrics": test_metrics,
        "coefficients": _coefficients(model),
    }


def _date_boundary(X: pd.DataFrame, boundary: str) -> str | None:
    if DATE_COLUMN not in X.columns:
        return None
    values = pd.to_datetime(X[DATE_COLUMN], errors="coerce")
    if values.isna().all():
        return None
    value = values.min() if boundary == "min" else values.max()
    return str(value.date())


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
