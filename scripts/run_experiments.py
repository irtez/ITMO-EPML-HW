"""Standalone script — run 20 experiments and compare results via MLflow.

This script demonstrates three MLflow integration patterns from the
``housing.tracking`` module:

1. **Decorator** (``@mlflow_run``) — used for the baseline dummy predictor.
2. **Context manager** (``ExperimentTracker``) — used for every real model.
3. **Utilities** (``compare_runs``, ``search_runs``, ``get_best_run``) —
   used to display a comparison table at the end.

Usage::

    # Make sure processed data exists first:
    poetry run dvc repro prepare

    # Then run experiments:
    poetry run python scripts/run_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from housing.features.build import get_feature_pipeline, split_features_target
from housing.models.train import compute_metrics
from housing.tracking import (
    ExperimentTracker,
    compare_runs,
    get_best_run,
    mlflow_run,
    search_runs,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "boston-housing-hw3"
TRAIN_CSV = Path("data/processed/train.csv")
VAL_SPLIT = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Experiment definitions — 20 runs across 6 algorithm families
# ---------------------------------------------------------------------------

ExperimentConfig = dict[str, Any]

EXPERIMENT_CONFIGS: list[ExperimentConfig] = [
    # ---- Linear family -------------------------------------------------------
    {
        "name": "LinearRegression",
        "model": LinearRegression(),
        "params": {},
        "family": "linear",
    },
    {
        "name": "Ridge_a0.001",
        "model": Ridge(alpha=0.001),
        "params": {"alpha": 0.001},
        "family": "linear",
    },
    {
        "name": "Ridge_a0.01",
        "model": Ridge(alpha=0.01),
        "params": {"alpha": 0.01},
        "family": "linear",
    },
    {
        "name": "Ridge_a0.1",
        "model": Ridge(alpha=0.1),
        "params": {"alpha": 0.1},
        "family": "linear",
    },
    {
        "name": "Ridge_a1.0",
        "model": Ridge(alpha=1.0),
        "params": {"alpha": 1.0},
        "family": "linear",
    },
    {
        "name": "Ridge_a10.0",
        "model": Ridge(alpha=10.0),
        "params": {"alpha": 10.0},
        "family": "linear",
    },
    {
        "name": "Lasso_a0.1",
        "model": Lasso(alpha=0.1),
        "params": {"alpha": 0.1},
        "family": "linear",
    },
    {
        "name": "Lasso_a1.0",
        "model": Lasso(alpha=1.0),
        "params": {"alpha": 1.0},
        "family": "linear",
    },
    {
        "name": "ElasticNet_a0.1_l0.5",
        "model": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "params": {"alpha": 0.1, "l1_ratio": 0.5},
        "family": "linear",
    },
    {
        "name": "ElasticNet_a0.5_l0.7",
        "model": ElasticNet(alpha=0.5, l1_ratio=0.7),
        "params": {"alpha": 0.5, "l1_ratio": 0.7},
        "family": "linear",
    },
    # ---- Tree family ---------------------------------------------------------
    {
        "name": "DecisionTree_d3",
        "model": DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
        "params": {"max_depth": 3},
        "family": "tree",
    },
    {
        "name": "DecisionTree_d5",
        "model": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "params": {"max_depth": 5},
        "family": "tree",
    },
    {
        "name": "DecisionTree_d10",
        "model": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "params": {"max_depth": 10},
        "family": "tree",
    },
    {
        "name": "DecisionTree_dNone",
        "model": DecisionTreeRegressor(max_depth=None, random_state=RANDOM_STATE),
        "params": {"max_depth": "None"},
        "family": "tree",
    },
    # ---- Ensemble family -----------------------------------------------------
    {
        "name": "RandomForest_n50_d5",
        "model": RandomForestRegressor(
            n_estimators=50, max_depth=5, random_state=RANDOM_STATE
        ),
        "params": {"n_estimators": 50, "max_depth": 5},
        "family": "ensemble",
    },
    {
        "name": "RandomForest_n100_d10",
        "model": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE
        ),
        "params": {"n_estimators": 100, "max_depth": 10},
        "family": "ensemble",
    },
    {
        "name": "RandomForest_n200",
        "model": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=RANDOM_STATE
        ),
        "params": {"n_estimators": 200, "max_depth": "None"},
        "family": "ensemble",
    },
    {
        "name": "GradientBoosting_n100_lr0.1",
        "model": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        "family": "ensemble",
    },
    {
        "name": "GradientBoosting_n200_lr0.05",
        "model": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        "family": "ensemble",
    },
    {
        "name": "ExtraTrees_n100_d10",
        "model": ExtraTreesRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE
        ),
        "params": {"n_estimators": 100, "max_depth": 10},
        "family": "ensemble",
    },
]


# ---------------------------------------------------------------------------
# Pattern 1: Decorator — baseline dummy mean predictor
# ---------------------------------------------------------------------------


@mlflow_run(
    experiment_name=EXPERIMENT_NAME,
    run_name="DummyMean_baseline",
    tags={"family": "baseline", "hw": "hw3"},
)
def log_baseline(y_train: pd.Series, y_val: pd.Series) -> None:
    """Log a trivial mean-predictor as an experiment baseline."""
    baseline_pred = np.full(len(y_val), float(y_train.mean()))
    metrics = compute_metrics(y_val, baseline_pred)
    mlflow.log_param("model_type", "DummyMean")
    mlflow.log_metrics({f"train_{k}": v for k, v in metrics.items()})
    mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})


# ---------------------------------------------------------------------------
# Pattern 2: Context manager — one real model per ExperimentTracker block
# ---------------------------------------------------------------------------


def run_one_experiment(
    cfg: ExperimentConfig,
    feature_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[str | None, float]:
    """Train one model inside an ExperimentTracker context."""
    model: BaseEstimator = cfg["model"]
    name: str = cfg["name"]
    params: dict[str, Any] = cfg["params"]
    family: str = cfg["family"]

    full_pipeline = Pipeline(
        [
            ("features", feature_pipeline),
            ("model", model),
        ]
    )

    with ExperimentTracker(
        experiment_name=EXPERIMENT_NAME,
        run_name=name,
        tags={"family": family, "hw": "hw3"},
    ) as tracker:
        tracker.log_param("model_type", name)
        if params:
            tracker.log_params(params)

        full_pipeline.fit(X_train, y_train)

        train_preds: np.ndarray[Any, np.dtype[np.float64]] = full_pipeline.predict(
            X_train
        )
        val_preds: np.ndarray[Any, np.dtype[np.float64]] = full_pipeline.predict(X_val)

        train_metrics = compute_metrics(y_train, train_preds)
        val_metrics = compute_metrics(y_val, val_preds)

        tracker.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        tracker.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        mlflow.sklearn.log_model(full_pipeline, artifact_path="model")

        run_id = tracker.run_id
        val_rmse = float(val_metrics["rmse"])

    print(
        f"  [{family:8s}] {name:<35s}  val_rmse={val_rmse:.4f}  "
        f"val_r2={val_metrics['r2']:.4f}"
    )
    return run_id, val_rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Load train data, run all experiments, print comparison table."""
    # ------------------------------------------------------------------
    # Validate data availability
    # ------------------------------------------------------------------
    if not TRAIN_CSV.exists():
        print(
            f"ERROR: {TRAIN_CSV} not found.\nRun `poetry run dvc repro prepare` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load params and data
    # ------------------------------------------------------------------
    raw_params: dict[str, Any] = yaml.safe_load(
        Path("params.yaml").read_text(encoding="utf-8")
    )
    exp_params: dict[str, Any] = raw_params.get("experiments", {})
    val_size: float = float(exp_params.get("val_size", VAL_SPLIT))
    random_state: int = int(exp_params.get("random_state", RANDOM_STATE))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    train_df = pd.read_csv(TRAIN_CSV)
    X_all, y_all = split_features_target(train_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=val_size,
        random_state=random_state,
    )

    print(
        f"Dataset: {len(X_train)} train / {len(X_val)} val samples, "
        f"{X_train.shape[1]} features"
    )
    print(f"MLflow experiment: {EXPERIMENT_NAME!r}\n")

    # ------------------------------------------------------------------
    # Pattern 1: baseline via decorator
    # ------------------------------------------------------------------
    print("Logging baseline (decorator pattern) …")
    log_baseline(y_train, y_val)

    # ------------------------------------------------------------------
    # Pattern 2: all real models via context manager
    # ------------------------------------------------------------------
    print(f"\nRunning {len(EXPERIMENT_CONFIGS)} experiments …\n")
    results: list[tuple[str | None, str, float]] = []
    for cfg in EXPERIMENT_CONFIGS:
        run_id, val_rmse = run_one_experiment(
            cfg=cfg,
            feature_pipeline=get_feature_pipeline(),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        results.append((run_id, str(cfg["name"]), val_rmse))

    # ------------------------------------------------------------------
    # Pattern 3: utilities — compare and search
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON (sorted by val_rmse ↑)")
    print("=" * 70)
    comparison_df = compare_runs(EXPERIMENT_NAME)
    if not comparison_df.empty:
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(comparison_df.to_string(index=False))

    print("\n--- Runs with val_r2 > 0.85 ---")
    high_r2 = search_runs(
        EXPERIMENT_NAME,
        filter_string="metrics.val_r2 > 0.85",
    )
    if not high_r2.empty:
        cols = ["run_name", "model_type", "metric_val_rmse", "metric_val_r2"]
        print(high_r2[cols].to_string(index=False))
    else:
        print("  (none)")

    print("\n--- Best run ---")
    best = get_best_run(EXPERIMENT_NAME, metric="val_rmse", mode="min")
    if best is not None:
        print(
            f"  run_id   : {best.info.run_id[:8]}\n"
            f"  run_name : {best.info.run_name}\n"
            f"  val_rmse : {best.data.metrics.get('val_rmse', float('nan')):.4f}\n"
            f"  val_r2   : {best.data.metrics.get('val_r2', float('nan')):.4f}"
        )

    print("\nDone. Open MLflow UI with:")
    print("  poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")


if __name__ == "__main__":
    main()
