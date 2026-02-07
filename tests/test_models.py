"""Tests for housing.models.train — pure functions only, no MLflow side effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing.features.build import split_features_target
from housing.models.train import compute_metrics


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
