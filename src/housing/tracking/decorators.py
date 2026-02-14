"""Decorators for automatic MLflow experiment tracking."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import mlflow

P = ParamSpec("P")
R = TypeVar("R")


def mlflow_run(
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a function inside an MLflow run.

    Sets the experiment (if provided), starts a run before calling the
    function, and ends the run when the function returns or raises.

    Args:
        experiment_name: MLflow experiment to use. If *None*, the currently
            active experiment is kept unchanged.
        run_name: Human-readable name shown in the MLflow UI.
        tags: Key-value tags attached to the run.

    Returns:
        Decorator that transparently wraps the target function.

    Example::

        @mlflow_run(experiment_name="my-exp", run_name="baseline")
        def train(params: dict[str, float]) -> float:
            mlflow.log_params(params)
            ...
            mlflow.log_metric("rmse", rmse)
            return rmse
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if experiment_name is not None:
                mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name):
                if tags is not None:
                    mlflow.set_tags(tags)
                return func(*args, **kwargs)

        return wrapper

    return decorator
