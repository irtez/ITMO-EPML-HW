"""Model training and evaluation utilities for Boston Housing."""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray[Any, np.dtype[np.float64]],
) -> dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with keys ``rmse``, ``mae``, ``r2``.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_log_model(
    model_name: str,
    model: BaseEstimator,
    feature_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    extra_params: dict[str, Any] | None = None,
) -> tuple[str, float]:
    """Train a full pipeline, log everything to MLflow, return run metadata.

    Builds a combined pipeline: ``feature_pipeline`` → ``model``, fits it on
    training data, evaluates on both train and validation sets, and logs all
    parameters, metrics, and the sklearn model artifact to the active MLflow
    experiment.

    Args:
        model_name: Human-readable identifier used as the MLflow run name.
        model: Unfitted sklearn-compatible estimator.
        feature_pipeline: Unfitted sklearn Pipeline for preprocessing.
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_val: Validation feature matrix.
        y_val: Validation target vector.
        extra_params: Additional hyperparameters to log to MLflow.

    Returns:
        Tuple ``(run_id, val_rmse)`` where ``run_id`` is the MLflow run ID and
        ``val_rmse`` is the validation RMSE.
    """
    full_pipeline = Pipeline(
        [
            ("features", feature_pipeline),
            ("model", model),
        ]
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_type", model_name)
        if extra_params:
            mlflow.log_params(extra_params)

        full_pipeline.fit(X_train, y_train)

        train_preds: np.ndarray[Any, np.dtype[np.float64]] = full_pipeline.predict(
            X_train
        )
        val_preds: np.ndarray[Any, np.dtype[np.float64]] = full_pipeline.predict(X_val)

        train_metrics = compute_metrics(y_train, train_preds)
        val_metrics = compute_metrics(y_val, val_preds)

        mlflow.log_metric("train_rmse", train_metrics["rmse"])
        mlflow.log_metric("train_mae", train_metrics["mae"])
        mlflow.log_metric("train_r2", train_metrics["r2"])
        mlflow.log_metric("val_rmse", val_metrics["rmse"])
        mlflow.log_metric("val_mae", val_metrics["mae"])
        mlflow.log_metric("val_r2", val_metrics["r2"])

        mlflow.sklearn.log_model(full_pipeline, artifact_path="model")

        run_id: str = run.info.run_id
        return run_id, val_metrics["rmse"]
