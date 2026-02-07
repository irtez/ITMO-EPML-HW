"""Data loading utilities for the housing dataset."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[3] / "data"

# Boston Housing Dataset column names (UCI / StatLib)
BOSTON_COLUMNS = [
    "crim",  # per capita crime rate by town
    "zn",  # proportion of residential land zoned for large lots
    "indus",  # proportion of non-retail business acres per town
    "chas",  # Charles River dummy variable (1 if borders river)
    "nox",  # nitric oxides concentration (parts per 10 million)
    "rm",  # average number of rooms per dwelling
    "age",  # proportion of owner-occupied units built before 1940
    "dis",  # weighted distance to five Boston employment centres
    "rad",  # index of accessibility to radial highways
    "tax",  # full-value property-tax rate per $10,000
    "ptratio",  # pupil-teacher ratio by town
    "b",  # 1000(Bk - 0.63)^2, Bk = proportion of Black residents
    "lstat",  # % lower status of the population
    "medv",  # target: median value of owner-occupied homes ($1000s)
]


def load_raw_housing(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Boston housing dataset.

    The file has no header and uses whitespace as separator.

    Args:
        path: Optional path override. Defaults to data/raw/housing.csv.

    Returns:
        DataFrame with raw housing data.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if path is None:
        path = DATA_DIR / "raw" / "housing.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=BOSTON_COLUMNS)
