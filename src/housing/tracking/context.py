"""Context manager for MLflow experiment tracking."""

from __future__ import annotations

from types import TracebackType
from typing import Any

import mlflow


class ExperimentTracker:
    """Context manager that wraps a single MLflow run.

    Handles experiment selection, run lifecycle, and tag attachment so that
    experiment code can focus on training logic rather than MLflow boilerplate.

    Args:
        experiment_name: Name of the MLflow experiment to use (created if
            it does not exist yet).
        run_name: Human-readable name for this run shown in the UI.
        tags: Optional key-value tags attached to the run on entry.

    Example::

        with ExperimentTracker("my-exp", run_name="ridge-a0.1",
                               tags={"family": "linear"}) as tracker:
            tracker.log_params({"alpha": 0.1})
            model.fit(X_train, y_train)
            tracker.log_metric("val_rmse", rmse)
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tags = tags
        self._run_id: str | None = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> ExperimentTracker:
        mlflow.set_experiment(self._experiment_name)
        mlflow.start_run(run_name=self._run_name)
        active = mlflow.active_run()
        if active is not None:
            self._run_id = active.info.run_id
        if self._tags is not None:
            mlflow.set_tags(self._tags)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        mlflow.end_run()

    # ------------------------------------------------------------------
    # Logging helpers (thin wrappers around mlflow.log_*)
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters."""
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dictionary of scalar metrics."""
        mlflow.log_metrics(metrics, step=step)

    def set_tag(self, key: str, value: str) -> None:
        """Set a single tag on the active run."""
        mlflow.set_tag(key, value)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str | None:
        """Return the MLflow run ID of the current run, or *None*."""
        return self._run_id
