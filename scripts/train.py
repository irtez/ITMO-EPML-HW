"""DVC stage: train one model variant configured via Hydra."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from housing.config import validate_train_cfg
from housing.features.build import get_feature_pipeline, split_features_target
from housing.models.train import train_and_log_model
from housing.pipeline.monitoring import monitored_stage


def _build_model(cfg: DictConfig) -> tuple[Any, dict[str, Any]]:
    model_kind = str(cfg.model.kind)
    params = dict(cfg.model.params)
    random_state = int(cfg.train.random_state)

    if model_kind == "linear_regression":
        return LinearRegression(), params
    if model_kind == "ridge":
        alpha = float(params.get("alpha", 1.0))
        return Ridge(alpha=alpha), {"alpha": alpha}
    if model_kind == "random_forest":
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = int(params.get("max_depth", 10))
        return (
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            ),
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
            },
        )

    raise ValueError(f"Unsupported model kind: {model_kind}")


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Train the selected model and save model artifact + metrics JSON."""
    validate_train_cfg(cfg)

    with monitored_stage(cfg, f"train:{cfg.model.name}"):
        mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
        mlflow.set_experiment(str(cfg.mlflow.experiment_name))

        train_df = pd.read_csv(str(cfg.train.input.train_path))
        X_all, y_all = split_features_target(train_df)

        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all,
            test_size=float(cfg.train.val_size),
            random_state=int(cfg.train.random_state),
        )

        model, extra_params = _build_model(cfg)

        run_id, val_rmse = train_and_log_model(
            model_name=str(cfg.model.name),
            model=model,
            feature_pipeline=get_feature_pipeline(),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            extra_params=extra_params if extra_params else None,
        )

        run = MlflowClient().get_run(run_id)
        metrics_json = {
            "model_name": str(cfg.model.name),
            "run_id": run_id,
            "train_rmse": float(run.data.metrics["train_rmse"]),
            "train_mae": float(run.data.metrics["train_mae"]),
            "train_r2": float(run.data.metrics["train_r2"]),
            "val_rmse": float(val_rmse),
            "val_mae": float(run.data.metrics["val_mae"]),
            "val_r2": float(run.data.metrics["val_r2"]),
        }

        model_uri = f"runs:/{run_id}/model"
        trained_pipeline = mlflow.sklearn.load_model(model_uri)

        model_path = Path(str(cfg.train.output.model_path))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_pipeline, model_path)  # nosec B301

        metrics_path = Path(str(cfg.train.output.metrics_path))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

        print(
            f"Model {cfg.model.name} trained: val_rmse={val_rmse:.4f}, "
            f"run_id={run_id}, saved={model_path}"
        )


if __name__ == "__main__":
    main()
