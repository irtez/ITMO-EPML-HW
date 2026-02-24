"""Tests for data loading utilities."""

from pathlib import Path

import pytest

from housing.data.load import BOSTON_COLUMNS, load_raw_housing


def test_load_raw_housing_returns_dataframe(housing_csv: Path) -> None:
    df = load_raw_housing(housing_csv)
    assert len(df) > 0


def test_load_raw_housing_columns(housing_csv: Path) -> None:
    df = load_raw_housing(housing_csv)
    assert set(df.columns) == set(BOSTON_COLUMNS)


def test_load_raw_housing_target_column_exists(housing_csv: Path) -> None:
    df = load_raw_housing(housing_csv)
    assert "medv" in df.columns


def test_load_raw_housing_custom_path_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_housing(Path("/nonexistent/path/housing.csv"))
