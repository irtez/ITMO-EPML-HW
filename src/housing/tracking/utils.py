"""Utilities for querying and comparing MLflow experiments."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# Default metrics shown in comparison tables
_DEFAULT_METRICS = ["val_rmse", "val_mae", "val_r2", "train_rmse", "train_r2"]


def get_best_run(
    experiment_name: str,
    metric: str = "val_rmse",
    mode: str = "min",
) -> mlflow.entities.Run | None:
    """Return the single best run from an experiment.

    Args:
        experiment_name: Name of the MLflow experiment to search.
        metric: Metric key to optimise (e.g. ``"val_rmse"``).
        mode: ``"min"`` to minimise the metric, ``"max"`` to maximise.

    Returns:
        The best :class:`mlflow.entities.Run`, or *None* if the experiment
        does not exist or has no finished runs with the requested metric.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    order_dir = "ASC" if mode == "min" else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} > -1e15",
        order_by=[f"metrics.{metric} {order_dir}"],
        max_results=1,
    )
    return runs[0] if runs else None


def compare_runs(
    experiment_name: str,
    metrics: list[str] | None = None,
    max_results: int = 100,
) -> pd.DataFrame:
    """Return a tidy DataFrame comparing all runs in an experiment.

    Columns: ``run_id`` (short), ``run_name``, ``model_type``, then one
    column per requested metric.

    Args:
        experiment_name: Name of the MLflow experiment.
        metrics: Metric keys to include. Defaults to
            ``["val_rmse", "val_mae", "val_r2", "train_rmse", "train_r2"]``.
        max_results: Maximum number of runs to retrieve (ordered by
            ``val_rmse`` ascending).

    Returns:
        A :class:`pandas.DataFrame` with one row per run, sorted by
        ``val_rmse`` ascending.  Empty DataFrame if the experiment does not
        exist.
    """
    metrics_to_show = metrics if metrics is not None else _DEFAULT_METRICS

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_rmse ASC"],
        max_results=max_results,
    )

    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {
            "run_id": run.info.run_id[:8],
            "run_name": run.info.run_name or "",
            "model_type": run.data.params.get("model_type", ""),
        }
        for m in metrics_to_show:
            row[m] = run.data.metrics.get(m)
        rows.append(row)

    return pd.DataFrame(rows)


def search_runs(
    experiment_name: str,
    filter_string: str = "",
    max_results: int = 100,
    order_by: list[str] | None = None,
) -> pd.DataFrame:
    """Search runs using an MLflow filter expression.

    Args:
        experiment_name: Name of the MLflow experiment.
        filter_string: MLflow filter expression, e.g.
            ``"metrics.val_r2 > 0.85"`` or
            ``"params.model_type = 'RandomForest'"``
        max_results: Maximum number of matching runs to return.
        order_by: List of ordering clauses, e.g.
            ``["metrics.val_rmse ASC"]``.  Defaults to ordering by
            ``val_rmse`` ascending.

    Returns:
        A :class:`pandas.DataFrame` where each row is a matching run.
        Parameter columns have their original names; metric columns are
        prefixed with ``metric_``.  Empty DataFrame if no experiment found.

    Example::

        df = search_runs(
            "boston-housing-hw3",
            filter_string="metrics.val_r2 > 0.85",
        )
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=order_by or ["metrics.val_rmse ASC"],
        max_results=max_results,
    )

    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {
            "run_id": run.info.run_id[:8],
            "run_name": run.info.run_name or "",
        }
        row.update(run.data.params)
        row.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
        rows.append(row)

    return pd.DataFrame(rows)
