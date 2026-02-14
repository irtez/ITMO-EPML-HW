"""DVC stage 2 — train models, log to MLflow, register the best one."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from housing.features.build import get_feature_pipeline, split_features_target
from housing.models.train import train_and_log_model


def main() -> None:
    """Train all configured models, pick the best by val RMSE, register it."""
    raw_params: dict[str, Any] = yaml.safe_load(
        Path("params.yaml").read_text(encoding="utf-8")
    )
    params: dict[str, Any] = raw_params["train"]
    random_state: int = int(params["random_state"])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("boston-housing")

    train_df = pd.read_csv("data/processed/train.csv")
    X_train, y_train = split_features_target(train_df)

    ridge_params: dict[str, Any] = dict(params["models"].get("ridge", {}))
    rf_params: dict[str, Any] = dict(params["models"].get("random_forest", {}))

    model_configs: dict[str, tuple[Any, dict[str, Any]]] = {
        "linear_regression": (LinearRegression(), {}),
        "ridge": (
            Ridge(alpha=float(ridge_params.get("alpha", 1.0))),
            ridge_params,
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=int(rf_params.get("n_estimators", 100)),
                max_depth=int(rf_params.get("max_depth", 10)),
                random_state=random_state,
            ),
            rf_params,
        ),
    }

    results: list[tuple[str, str, float]] = []
    for name, (model, extra) in model_configs.items():
        run_id, val_rmse = train_and_log_model(
            model_name=name,
            model=model,
            feature_pipeline=get_feature_pipeline(),
            X_train=X_train,
            y_train=y_train,
            X_val=X_train,
            y_val=y_train,
            extra_params=extra if extra else None,
        )
        results.append((name, run_id, val_rmse))
        print(f"  {name}: train_rmse={val_rmse:.4f}  run_id={run_id}")

    best_name, best_run_id, best_rmse = min(results, key=lambda t: t[2])
    print(f"\nBest model: {best_name}  (rmse={best_rmse:.4f})")

    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="housing-price-predictor")

    best_pipeline = mlflow.sklearn.load_model(model_uri)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_pipeline, "models/best_model.joblib")  # nosec B301

    Path("metrics").mkdir(exist_ok=True)
    train_metrics = {
        "best_model": best_name,
        "best_train_rmse": best_rmse,
        "all_models": {n: {"train_rmse": r} for n, _, r in results},
    }
    Path("metrics/train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2), encoding="utf-8"
    )
    print("Saved best_model.joblib and metrics/train_metrics.json")


if __name__ == "__main__":
    main()
