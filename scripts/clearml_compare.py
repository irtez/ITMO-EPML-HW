"""Export ClearML experiment/model comparisons to CSV tables."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from housing.config import validate_clearml_cfg
from housing.tracking import compare_clearml_experiments, compare_registered_models


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Query ClearML server and save comparison artifacts to metrics/."""
    validate_clearml_cfg(cfg)
    if not bool(cfg.clearml.enabled):
        print("ClearML disabled (clearml.enabled=false), skipping comparison export.")
        return

    experiments_df = compare_clearml_experiments(
        project_name=str(cfg.clearml.project_name),
        metric_name="val_rmse",
    )
    models_df = compare_registered_models(
        model_name=str(cfg.clearml.model_name),
        project_name=str(cfg.clearml.project_name),
    )

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    exp_path = metrics_dir / "clearml_experiments.csv"
    mdl_path = metrics_dir / "clearml_models.csv"
    experiments_df.to_csv(exp_path, index=False)
    models_df.to_csv(mdl_path, index=False)

    print(f"Saved experiments comparison: {exp_path} ({len(experiments_df)} rows)")
    print(f"Saved model comparison: {mdl_path} ({len(models_df)} rows)")


if __name__ == "__main__":
    main()
