"""DVC stage 3 — evaluate the best saved model on the held-out test set."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from housing.features.build import split_features_target
from housing.models.train import compute_metrics


def main() -> None:
    """Load best model and test data, compute metrics, save results."""
    pipeline = joblib.load("models/best_model.joblib")  # nosec B301

    test_df = pd.read_csv("data/processed/test.csv")
    X_test, y_test = split_features_target(test_df)

    preds = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"Test metrics — RMSE: {metrics['rmse']:.4f}  "
        f"MAE: {metrics['mae']:.4f}  R²: {metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
