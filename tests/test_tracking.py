"""Tests for housing.tracking — decorators, context manager, and utilities."""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest

from housing.tracking import ExperimentTracker, compare_runs, get_best_run, mlflow_run
from housing.tracking.utils import search_runs

# ---------------------------------------------------------------------------
# Fixture: isolate every test in its own temporary MLflow store
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
    """Point MLflow at a fresh temp directory for each test."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield  # type: ignore[misc]
    # Clean up active run if any test left one open
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


class TestMlflowRunDecorator:
    def test_decorator_starts_and_ends_run(self) -> None:
        """Wrapped function executes inside an MLflow run that is closed on exit."""
        captured: list[str] = []

        @mlflow_run(experiment_name="test-exp", run_name="test-run")
        def _inner() -> None:
            run = mlflow.active_run()
            assert run is not None
            captured.append(run.info.run_id)

        assert mlflow.active_run() is None
        _inner()
        assert mlflow.active_run() is None
        assert len(captured) == 1

    def test_decorator_sets_experiment(self) -> None:
        """Wrapped function is logged to the specified experiment."""

        @mlflow_run(experiment_name="my-experiment")
        def _inner() -> None:
            run = mlflow.active_run()
            assert run is not None
            exp = mlflow.get_experiment(run.info.experiment_id)
            assert exp is not None
            assert exp.name == "my-experiment"

        _inner()

    def test_decorator_sets_tags(self) -> None:
        """Tags provided to the decorator are attached to the run."""
        captured_run_id: list[str] = []

        @mlflow_run(
            experiment_name="tagged-exp",
            tags={"env": "test", "version": "1"},
        )
        def _inner() -> None:
            run = mlflow.active_run()
            assert run is not None
            captured_run_id.append(run.info.run_id)

        _inner()
        # Tags must be read via the client after the run finishes
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(captured_run_id[0])
        assert run.data.tags.get("env") == "test"
        assert run.data.tags.get("version") == "1"

    def test_decorator_preserves_return_value(self) -> None:
        """The decorated function's return value is passed through."""

        @mlflow_run(experiment_name="ret-exp")
        def _inner() -> int:
            return 42

        assert _inner() == 42

    def test_decorator_without_experiment_name(self) -> None:
        """Decorator without experiment_name does not override active experiment."""
        mlflow.set_experiment("pre-set-exp")

        @mlflow_run(run_name="no-exp-run")
        def _inner() -> None:
            run = mlflow.active_run()
            assert run is not None
            exp = mlflow.get_experiment(run.info.experiment_id)
            assert exp is not None
            assert exp.name == "pre-set-exp"

        _inner()


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


class TestExperimentTracker:
    def test_context_manager_starts_run(self) -> None:
        """Run is active inside the context block."""
        assert mlflow.active_run() is None
        with ExperimentTracker("ctx-exp"):
            assert mlflow.active_run() is not None
        assert mlflow.active_run() is None

    def test_context_manager_sets_experiment(self) -> None:
        """Experiment is set to the provided name."""
        with ExperimentTracker("ctx-exp-named"):
            run = mlflow.active_run()
            assert run is not None
            exp = mlflow.get_experiment(run.info.experiment_id)
            assert exp is not None
            assert exp.name == "ctx-exp-named"

    def test_context_manager_run_id_property(self) -> None:
        """run_id property returns the active run's ID."""
        with ExperimentTracker("ctx-exp-id") as tracker:
            assert tracker.run_id is not None
            active = mlflow.active_run()
            assert active is not None
            assert tracker.run_id == active.info.run_id

    def test_log_param_and_metric(self) -> None:
        """Parameters and metrics logged via tracker are stored in the run."""
        with ExperimentTracker("ctx-log-exp", run_name="log-run") as tracker:
            tracker.log_param("alpha", 0.5)
            tracker.log_metric("val_rmse", 3.14)
            run_id = tracker.run_id

        client = mlflow.tracking.MlflowClient()
        assert run_id is not None
        run = client.get_run(run_id)
        assert run.data.params.get("alpha") == "0.5"
        assert abs(run.data.metrics.get("val_rmse", 0.0) - 3.14) < 1e-6

    def test_log_params_and_metrics_dicts(self) -> None:
        """Batch logging of params and metrics works correctly."""
        with ExperimentTracker("ctx-batch-exp") as tracker:
            tracker.log_params({"a": 1, "b": "hello"})
            tracker.log_metrics({"m1": 1.0, "m2": 2.0})
            run_id = tracker.run_id

        client = mlflow.tracking.MlflowClient()
        assert run_id is not None
        run = client.get_run(run_id)
        assert run.data.params.get("a") == "1"
        assert abs(run.data.metrics.get("m1", 0.0) - 1.0) < 1e-6

    def test_set_tag(self) -> None:
        """Tags set via set_tag are attached to the run."""
        with ExperimentTracker("ctx-tag-exp") as tracker:
            tracker.set_tag("framework", "sklearn")
            run_id = tracker.run_id

        client = mlflow.tracking.MlflowClient()
        assert run_id is not None
        run = client.get_run(run_id)
        assert run.data.tags.get("framework") == "sklearn"

    def test_tags_on_entry(self) -> None:
        """Tags passed to constructor are set when entering the context."""
        with ExperimentTracker(
            "ctx-init-tags", tags={"hw": "hw3", "family": "linear"}
        ) as tracker:
            run_id = tracker.run_id

        client = mlflow.tracking.MlflowClient()
        assert run_id is not None
        run = client.get_run(run_id)
        assert run.data.tags.get("hw") == "hw3"


# ---------------------------------------------------------------------------
# Utilities tests
# ---------------------------------------------------------------------------


class TestUtils:
    @staticmethod
    def _seed_experiment(exp_name: str, n_runs: int = 3) -> list[str]:
        """Create n_runs test runs in exp_name and return their IDs."""
        mlflow.set_experiment(exp_name)
        run_ids: list[str] = []
        for i in range(n_runs):
            with mlflow.start_run(run_name=f"run_{i}") as run:
                mlflow.log_param("model_type", f"model_{i}")
                mlflow.log_metric("val_rmse", float(i + 1))
                mlflow.log_metric("val_mae", float(i + 0.5))
                mlflow.log_metric("val_r2", 1.0 - float(i) * 0.1)
                mlflow.log_metric("train_rmse", float(i + 0.8))
                mlflow.log_metric("train_r2", 1.0 - float(i) * 0.05)
                run_ids.append(run.info.run_id)
        return run_ids

    def test_get_best_run_returns_run(self) -> None:
        """get_best_run returns the run with the lowest val_rmse."""
        self._seed_experiment("utils-best-exp")
        best = get_best_run("utils-best-exp", metric="val_rmse", mode="min")
        assert best is not None
        assert best.data.metrics.get("val_rmse") == 1.0

    def test_get_best_run_missing_experiment(self) -> None:
        """get_best_run returns None for a non-existent experiment."""
        result = get_best_run("does-not-exist-xyz")
        assert result is None

    def test_get_best_run_mode_max(self) -> None:
        """get_best_run with mode='max' returns highest metric."""
        self._seed_experiment("utils-max-exp")
        best = get_best_run("utils-max-exp", metric="val_r2", mode="max")
        assert best is not None
        assert abs(best.data.metrics.get("val_r2", 0.0) - 1.0) < 1e-6

    def test_compare_runs_returns_dataframe(self) -> None:
        """compare_runs returns a non-empty DataFrame with expected columns."""
        self._seed_experiment("utils-compare-exp")
        df = compare_runs("utils-compare-exp")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "run_name" in df.columns
        assert "val_rmse" in df.columns

    def test_compare_runs_missing_experiment(self) -> None:
        """compare_runs returns empty DataFrame for non-existent experiment."""
        df = compare_runs("no-such-experiment-abc")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_compare_runs_sorted_by_val_rmse(self) -> None:
        """Rows in compare_runs result are sorted by val_rmse ascending."""
        self._seed_experiment("utils-sorted-exp", n_runs=3)
        df = compare_runs("utils-sorted-exp")
        rmse_values = df["val_rmse"].dropna().tolist()
        assert rmse_values == sorted(rmse_values)

    def test_search_runs_with_filter(self) -> None:
        """search_runs respects the filter_string."""
        self._seed_experiment("utils-filter-exp", n_runs=3)
        # Only run_0 has val_rmse == 1.0 (< 1.5)
        df = search_runs(
            "utils-filter-exp",
            filter_string="metrics.val_rmse < 1.5",
        )
        assert not df.empty
        assert all(df["metric_val_rmse"] < 1.5)

    def test_search_runs_missing_experiment(self) -> None:
        """search_runs returns empty DataFrame for non-existent experiment."""
        df = search_runs("no-such-exp-xyz")
        assert isinstance(df, pd.DataFrame)
        assert df.empty
