"""DVC stage: evaluate selected best model on held-out test data."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig

from housing.config import validate_evaluate_cfg
from housing.features.build import split_features_target
from housing.models.train import compute_metrics
from housing.pipeline.monitoring import monitored_stage
from housing.tracking import ClearMLTracker


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Load best model and test data, compute metrics, and save JSON output."""
    validate_evaluate_cfg(cfg)

    with monitored_stage(cfg, "evaluate"):
        with ClearMLTracker(cfg, stage="evaluate", task_type="testing") as cml:
            pipeline = joblib.load(str(cfg.evaluate.input.model_path))  # nosec B301

            test_df = pd.read_csv(str(cfg.evaluate.input.test_path))
            X_test, y_test = split_features_target(test_df)

            preds = pipeline.predict(X_test)
            metrics = compute_metrics(y_test, preds)

            output_path = Path(str(cfg.evaluate.output.metrics_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            cml.log_metrics(metrics, series="test")
            cml.upload_artifact("test_metrics_json", output_path)

            print(
                f"Test metrics - RMSE: {metrics['rmse']:.4f} "
                f"MAE: {metrics['mae']:.4f} R2: {metrics['r2']:.4f}"
            )


if __name__ == "__main__":
    main()
