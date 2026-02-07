"""DVC stage 1 — split raw data into train / test sets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sklearn.model_selection import train_test_split

from housing.data.load import load_raw_housing


def main() -> None:
    """Load raw data, split into train/test, and save to data/processed/."""
    params: dict[str, Any] = yaml.safe_load(
        Path("params.yaml").read_text(encoding="utf-8")
    )["prepare"]
    test_size: float = float(params["test_size"])
    random_state: int = int(params["random_state"])

    df = load_raw_housing()
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"Saved train ({len(train_df)} rows) and test ({len(test_df)} rows).")


if __name__ == "__main__":
    main()
