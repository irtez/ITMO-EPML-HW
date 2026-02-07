"""Tests for src/housing/features/build.py."""

from __future__ import annotations

import pandas as pd
import pytest

from housing.features.build import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_feature_pipeline,
    split_features_target,
)


def test_split_features_target_shapes(tiny_df: pd.DataFrame) -> None:
    X, y = split_features_target(tiny_df)
    assert X.shape == (len(tiny_df), len(FEATURE_COLUMNS))
    assert y.shape == (len(tiny_df),)


def test_split_features_target_column_names(tiny_df: pd.DataFrame) -> None:
    X, y = split_features_target(tiny_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN


def test_feature_pipeline_transforms(tiny_df: pd.DataFrame) -> None:
    X, _ = split_features_target(tiny_df)
    pipeline = get_feature_pipeline()
    X_scaled = pipeline.fit_transform(X)
    assert abs(float(X_scaled.mean())) < 0.1  # type: ignore[union-attr]


def test_feature_pipeline_returns_pipeline(tiny_df: pd.DataFrame) -> None:
    from sklearn.pipeline import Pipeline

    assert isinstance(get_feature_pipeline(), Pipeline)


@pytest.mark.parametrize("col", FEATURE_COLUMNS)
def test_feature_columns_present_in_data(tiny_df: pd.DataFrame, col: str) -> None:
    assert col in tiny_df.columns
