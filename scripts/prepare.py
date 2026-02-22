"""DVC stage: split raw data into train/test with Hydra config."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from housing.config import validate_prepare_cfg
from housing.data.load import load_raw_housing
from housing.pipeline.monitoring import monitored_stage
from housing.tracking import ClearMLTracker


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Load raw data, split into train/test, and save to configured paths."""
    validate_prepare_cfg(cfg)

    with monitored_stage(cfg, "prepare"):
        with ClearMLTracker(cfg, stage="prepare", task_type="data_processing") as cml:
            df = load_raw_housing()
            train_df, test_df = train_test_split(
                df,
                test_size=float(cfg.prepare.test_size),
                random_state=int(cfg.prepare.random_state),
            )

            train_path = Path(str(cfg.prepare.output.train_path))
            test_path = Path(str(cfg.prepare.output.test_path))

            train_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.parent.mkdir(parents=True, exist_ok=True)

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            cml.connect_dict(
                "prepare_params",
                {
                    "test_size": float(cfg.prepare.test_size),
                    "random_state": int(cfg.prepare.random_state),
                },
            )
            cml.log_metrics(
                {
                    "raw_rows": float(len(df)),
                    "train_rows": float(len(train_df)),
                    "test_rows": float(len(test_df)),
                },
                series="prepare",
            )
            cml.upload_artifact("train_split_csv", train_path)
            cml.upload_artifact("test_split_csv", test_path)

            print(f"Saved train ({len(train_df)} rows) to {train_path}")
            print(f"Saved test ({len(test_df)} rows) to {test_path}")


if __name__ == "__main__":
    main()
