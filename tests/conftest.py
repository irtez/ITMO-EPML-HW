"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def housing_csv(tmp_path: Path, tiny_df: pd.DataFrame) -> Path:
    """Write tiny_df as whitespace-separated CSV (no header) matching housing.csv format."""  # noqa: E501
    path = tmp_path / "housing.csv"
    tiny_df.to_csv(path, sep=" ", header=False, index=False)
    return path


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Return a 20-row synthetic DataFrame matching Boston Housing schema."""
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame(
        {
            "crim": rng.uniform(0.01, 1.0, n),
            "zn": rng.uniform(0.0, 100.0, n),
            "indus": rng.uniform(1.0, 25.0, n),
            "chas": rng.integers(0, 2, n).astype(float),
            "nox": rng.uniform(0.3, 0.9, n),
            "rm": rng.uniform(4.0, 9.0, n),
            "age": rng.uniform(10.0, 100.0, n),
            "dis": rng.uniform(1.0, 12.0, n),
            "rad": rng.integers(1, 24, n).astype(float),
            "tax": rng.uniform(100.0, 700.0, n),
            "ptratio": rng.uniform(12.0, 22.0, n),
            "b": rng.uniform(0.0, 400.0, n),
            "lstat": rng.uniform(1.0, 40.0, n),
            "medv": rng.uniform(5.0, 50.0, n),
        }
    )
