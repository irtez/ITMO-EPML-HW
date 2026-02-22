"""DVC stage: pick best model from candidate metrics and register it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig

from housing.config import validate_select_cfg
from housing.pipeline.monitoring import monitored_stage
from housing.tracking import ClearMLTracker


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Select model with minimal val_rmse and write aggregate train metrics."""
    validate_select_cfg(cfg)

    with monitored_stage(cfg, "select_best"):
        with ClearMLTracker(cfg, stage="select_best", task_type="optimizer") as cml:
            mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
            mlflow.set_experiment(str(cfg.mlflow.experiment_name))

            rows: list[dict[str, Any]] = []
            for candidate in cfg.selection.candidates:
                metrics_path = Path(str(candidate.metrics_path))
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                data["artifact_path"] = str(candidate.model_path)
                rows.append(data)

            best = min(rows, key=lambda row: float(row["val_rmse"]))

            source = Path(str(best["artifact_path"]))
            target = Path(str(cfg.selection.output.best_model_path))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

            model_uri = f"runs:/{best['run_id']}/model"
            mlflow.register_model(model_uri=model_uri, name=str(cfg.mlflow.model_name))

            summary = {
                "best_model": best["model_name"],
                "best_run_id": best["run_id"],
                "best_val_rmse": float(best["val_rmse"]),
                "all_models": {
                    row["model_name"]: {
                        "val_rmse": float(row["val_rmse"]),
                        "val_mae": float(row["val_mae"]),
                        "val_r2": float(row["val_r2"]),
                        "train_rmse": float(row["train_rmse"]),
                    }
                    for row in rows
                },
            }

            output_path = Path(str(cfg.selection.output.metrics_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            cml.connect_dict(
                "selection",
                {"candidates": [row["model_name"] for row in rows]},
            )
            cml.log_metrics(
                {"best_val_rmse": float(best["val_rmse"])},
                series="selection",
            )
            cml.upload_artifact("selection_summary_json", output_path)
            cml.upload_artifact("best_model_joblib", target)
            cml.register_model(
                model_path=target,
                model_name=f"{cfg.clearml.model_name}-best",
                metadata=summary,
            )

            print(
                f"Best model: {best['model_name']} "
                f"(val_rmse={float(best['val_rmse']):.4f})"
            )


if __name__ == "__main__":
    main()
