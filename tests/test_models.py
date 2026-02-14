"""Tests for housing.models.train — pure functions only, no MLflow side effects."""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from housing.features.build import get_feature_pipeline, split_features_target
from housing.models.train import compute_metrics, train_and_log_model


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
    """Point MLflow to a temporary SQLite backend for each test."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-models-exp")
    yield  # type: ignore[misc]
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(None)  # type: ignore[arg-type]


def test_compute_metrics_perfect_prediction(tiny_df: pd.DataFrame) -> None:
    _, y = split_features_target(tiny_df)
    metrics = compute_metrics(y, y.to_numpy())
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)


def test_compute_metrics_returns_expected_keys(tiny_df: pd.DataFrame) -> None:
    _, y = split_features_target(tiny_df)
    preds = np.full(len(y), float(y.mean()))
    metrics = compute_metrics(y, preds)
    assert set(metrics.keys()) == {"rmse", "mae", "r2"}


def test_compute_metrics_values_are_floats(tiny_df: pd.DataFrame) -> None:
    _, y = split_features_target(tiny_df)
    preds = np.full(len(y), float(y.mean()))
    metrics = compute_metrics(y, preds)
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"


def test_compute_metrics_constant_prediction_r2_not_positive(
    tiny_df: pd.DataFrame,
) -> None:
    _, y = split_features_target(tiny_df)
    preds = np.full(len(y), float(y.mean()))
    metrics = compute_metrics(y, preds)
    assert metrics["r2"] == pytest.approx(0.0, abs=1e-9)


def test_compute_metrics_rmse_nonnegative(tiny_df: pd.DataFrame) -> None:
    _, y = split_features_target(tiny_df)
    rng = np.random.default_rng(0)
    preds = rng.uniform(float(y.min()), float(y.max()), len(y))
    metrics = compute_metrics(y, preds)
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_train_and_log_model_logs_run_with_params_and_model(
    tiny_df: pd.DataFrame,
) -> None:
    X, y = split_features_target(tiny_df)
    run_id, val_rmse = train_and_log_model(
        model_name="linear_regression",
        model=LinearRegression(),
        feature_pipeline=get_feature_pipeline(),
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y,
        extra_params={"alpha": 0.1},
    )

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    assert isinstance(run_id, str)
    assert run_id
    assert isinstance(val_rmse, float)
    assert "train_rmse" in run.data.metrics
    assert "val_rmse" in run.data.metrics
    assert run.data.params.get("model_type") == "linear_regression"
    assert run.data.params.get("alpha") == "0.1"
