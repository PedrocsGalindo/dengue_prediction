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
from sklearn.ensemble import RandomForestRegressor  # Alterado aqui
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dengue_prediction.settings import DATA_DIR

# Importamos as funções auxiliares do pipeline de linear regression para não repetir código
# Se preferir, você pode copiar as funções _prepare_features e _chronological_train_test_split 
# que você já tem no outro arquivo.
from ..linear_regression.nodes import (
    _prepare_features, 
    _chronological_train_test_split,
    _regression_metrics,
    _mean_fold_metrics
)

def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Train a chronological Random Forest regressor for dengue cases."""
    config = _resolve_rf_params(params)
    X_model = _prepare_features(X)
    y_model = pd.Series(y).reset_index(drop=True)

    X_train, X_test, y_train, y_test = _chronological_train_test_split(
        X_model,
        y_model,
        test_size=config["test_size"],
    )

    # Cross validation específico para RF
    fold_metrics = _cross_validate_rf(
        X_train,
        y_train,
        config=config
    )

    # Build e fit do modelo final
    model = _build_rf_model(config)
    
    started_at = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - started_at

    y_pred = model.predict(X_test)
    test_metrics = _regression_metrics(y_test, y_pred)
    
    predictions = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": y_pred,
        "residual": y_test.reset_index(drop=True) - y_pred,
    })

    report = _build_rf_report(
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
    
    _save_rf_artifacts(model, report, predictions)

    return model, report, predictions

def _resolve_rf_params(params: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(params or {})
    return {
        "test_size": float(params.get("test_size", 0.2)),
        "n_splits": int(params.get("n_splits", 5)),
        "n_estimators": int(params.get("n_estimators", 100)),
        "max_depth": params.get("max_depth", None), # Pode ser int ou None
        "random_state": int(params.get("random_state", 42)),
        "scale_features": bool(params.get("scale_features", False)), # RF geralmente não precisa de escala
    }

def _build_rf_model(config: dict[str, Any]) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if config["scale_features"]:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), make_column_selector(dtype_include=object)),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=config["random_state"],
            n_jobs=-1 # Usa todos os cores do Lightning AI para treinar mais rápido
        )),
    ])

def _cross_validate_rf(X_train, y_train, config) -> list[dict[str, float]]:
    max_splits = max(0, len(X_train) - 1)
    fold_metrics = []
    
    tscv = TimeSeriesSplit(n_splits=min(config["n_splits"], max_splits))
    
    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X_train), start=1):
        fold_model = _build_rf_model(config)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_pred = fold_model.predict(X_train.iloc[valid_idx])
        metrics = _regression_metrics(y_train.iloc[valid_idx], fold_pred)
        metrics["fold"] = float(fold)
        fold_metrics.append(metrics)
    return fold_metrics

def _get_feature_importance(model: Pipeline) -> list[dict[str, Any]]:
    """RF não tem coeficientes, mas tem feature importance."""
    rf_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]
    
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(len(rf_model.feature_importances_))]

    importances = [
        {
            "feature": str(name),
            "importance": float(imp),
        }
        for name, imp in zip(feature_names, rf_model.feature_importances_)
    ]
    return sorted(importances, key=lambda x: x["importance"], reverse=True)

def _build_rf_report(model, config, X_train, X_test, y_train, y_test, fold_metrics, test_metrics, training_time) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "metadata": {
            "backend": "scikit-learn",
            "model_name": "RandomForestRegressor",
            "task_type": "regression",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_time_seconds": float(training_time),
        },
        "methodology": {
            "split_strategy": "chronological holdout",
            "validation_strategy": "TimeSeriesSplit",
            "params": config,
        },
        "data": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "cross_validation": {
            "fold_metrics": fold_metrics,
            "mean_metrics": _mean_fold_metrics(fold_metrics),
        },
        "test_metrics": test_metrics,
        "feature_importances": _get_feature_importance(model),
    }

def _save_rf_artifacts(model, report, predictions):
    run_id = report["metadata"]["run_id"]
    output_dir = DATA_DIR / "results" / "random_forest" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "random_forest.joblib")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)