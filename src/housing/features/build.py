"""Feature engineering utilities for the Boston Housing dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: list[str] = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "b",
    "lstat",
]
TARGET_COLUMN: str = "medv"


def get_feature_pipeline() -> Pipeline:
    """Return an unfitted sklearn Pipeline that scales all features.

    Returns:
        Pipeline with a single StandardScaler step.
    """
    return Pipeline([("scaler", StandardScaler())])


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y.

    Args:
        df: DataFrame containing all Boston Housing columns including the target.

    Returns:
        Tuple (X, y) where X contains FEATURE_COLUMNS and y contains TARGET_COLUMN.
    """
    X: pd.DataFrame = df[FEATURE_COLUMNS]
    y: pd.Series = df[TARGET_COLUMN]
    return X, y
