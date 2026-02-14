"""Reproducibility checks for deterministic training config."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from housing.features.build import get_feature_pipeline, split_features_target
from housing.models.train import compute_metrics


def _run_once(df: pd.DataFrame) -> float:
    X, y = split_features_target(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )
    pipeline = Pipeline(
        [
            ("features", get_feature_pipeline()),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds: np.ndarray = pipeline.predict(X_val)
    metrics = compute_metrics(y_val, preds)
    return float(metrics["rmse"])


def test_random_forest_rmse_is_reproducible(tiny_df: pd.DataFrame) -> None:
    rmse_1 = _run_once(tiny_df)
    rmse_2 = _run_once(tiny_df)
    assert rmse_1 == rmse_2
