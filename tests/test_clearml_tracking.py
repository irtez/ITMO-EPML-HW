"""Tests for ClearML integration helpers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf

from housing.tracking.clearml import (
    ClearMLTracker,
    ClearMLUnavailableError,
    compare_clearml_experiments,
    compare_registered_models,
)


def _cfg(enabled: bool = True) -> object:
    return OmegaConf.create(
        {
            "clearml": {
                "enabled": enabled,
                "project_name": "boston-housing-hw5",
                "task_prefix": "housing",
                "model_name": "housing-model",
                "output_uri": "",
                "tags": ["hw5"],
                "server": {
                    "api_host": "http://localhost:8008",
                    "web_host": "http://localhost:8080",
                    "files_host": "http://localhost:8081",
                },
                "auth": {
                    "access_key_env": "CLEARML_API_ACCESS_KEY",
                    "secret_key_env": "CLEARML_API_SECRET_KEY",
                },
                "pipeline": {
                    "name": "housing-hw5",
                    "execution_queue": "default",
                    "run_locally": True,
                    "schedule_cron": "0 6 * * *",
                },
            }
        }
    )


class _FakeLogger:
    def __init__(self) -> None:
        self.scalars: list[dict[str, Any]] = []
        self.messages: list[str] = []

    def report_scalar(
        self,
        title: str,
        series: str,
        value: float,
        iteration: int,
    ) -> None:
        self.scalars.append(
            {
                "title": title,
                "series": series,
                "value": value,
                "iteration": iteration,
            }
        )

    def report_text(self, message: str) -> None:
        self.messages.append(message)


class _FakeTask:
    TaskTypes = types.SimpleNamespace(
        training="training",
        data_processing="data_processing",
        testing="testing",
        optimizer="optimizer",
    )

    created: list[_FakeTask] = []
    listed: list[Any] = []
    last_project: str | None = None

    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.id = f"task-{len(_FakeTask.created)}"
        self.name = str(kwargs.get("task_name", ""))
        self.status = "completed"
        self.started = "2026-02-24T10:00:00+00:00"
        self.completed = "2026-02-24T10:02:00+00:00"
        self.tags: list[str] = []
        self.connected: dict[str, Any] = {}
        self.configurations: dict[str, Any] = {}
        self.artifacts: list[tuple[str, Any]] = []
        self.logger = _FakeLogger()
        self.closed = False
        self.scalar_summary: dict[str, Any] = {
            "val_rmse": {"main": {"last": 1.0}},
        }

    @classmethod
    def init(cls, **kwargs: Any) -> _FakeTask:
        task = _FakeTask(kwargs)
        cls.created.append(task)
        return task

    @classmethod
    def get_tasks(cls, project_name: str) -> list[Any]:
        cls.last_project = project_name
        return cls.listed

    def add_tags(self, tags: list[str]) -> None:
        self.tags.extend(tags)

    def connect(self, payload: dict[str, Any], name: str) -> None:
        self.connected[name] = payload

    def connect_configuration(self, name: str, configuration: dict[str, Any]) -> None:
        self.configurations[name] = configuration

    def get_logger(self) -> _FakeLogger:
        return self.logger

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        self.artifacts.append((name, artifact_object))

    def close(self) -> None:
        self.closed = True

    def get_last_scalar_metrics(self) -> dict[str, Any]:
        return self.scalar_summary


class _FakeOutputModel:
    created: list[_FakeOutputModel] = []

    def __init__(self, task: _FakeTask, name: str) -> None:
        self.task = task
        self.name = name
        self.id = f"model-{len(_FakeOutputModel.created)}"
        self.weights_args: dict[str, Any] = {}
        self.design: dict[str, Any] = {}
        self.comment = ""
        self.metadata_items: dict[str, dict[str, Any]] = {}
        _FakeOutputModel.created.append(self)

    def update_weights(self, **kwargs: Any) -> None:
        self.weights_args = kwargs

    def update_design(
        self,
        config_text: str | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> None:
        del config_text
        self.design = config_dict or {}

    def set_metadata(self, key: str, value: str, v_type: str | None = None) -> None:
        self.metadata_items[key] = {"value": value, "type": v_type}


class _FakeRegistryModel:
    def __init__(
        self,
        model_id: str,
        name: str,
        uri: str,
        framework: str,
        published: str,
        task: str,
        tags: list[str],
    ) -> None:
        self.id = model_id
        self.name = name
        self.uri = uri
        self.framework = framework
        self.published = published
        self.task = task
        self.tags = tags


class _FakeModel:
    queried_with: dict[str, Any] = {}
    listed: list[_FakeRegistryModel] = []

    @classmethod
    def query_models(cls, **kwargs: Any) -> list[_FakeRegistryModel]:
        cls.queried_with = kwargs
        return cls.listed


def _install_fake_clearml(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("clearml")
    module.Task = _FakeTask  # type: ignore[attr-defined]
    module.OutputModel = _FakeOutputModel  # type: ignore[attr-defined]
    module.Model = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clearml", module)


def _reset_fakes() -> None:
    _FakeTask.created = []
    _FakeTask.listed = []
    _FakeTask.last_project = None
    _FakeOutputModel.created = []
    _FakeModel.queried_with = {}
    _FakeModel.listed = []


def test_clearml_tracker_disabled_is_noop() -> None:
    cfg = _cfg(enabled=False)
    tracker = ClearMLTracker(cfg, stage="prepare")  # type: ignore[arg-type]

    with tracker:
        tracker.connect_dict("params", {"a": 1})
        tracker.log_metrics({"rmse": 1.0})
        tracker.upload_artifact("x", {"k": "v"})

    assert tracker.task_id is None


def test_clearml_tracker_raises_if_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(enabled=True)

    def _raise(name: str, package: str | None = None) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _raise)

    with pytest.raises(ClearMLUnavailableError):
        with ClearMLTracker(cfg, stage="prepare"):  # type: ignore[arg-type]
            pass


def test_clearml_tracker_logs_metrics_and_registers_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _reset_fakes()
    _install_fake_clearml(monkeypatch)
    cfg = _cfg(enabled=True)

    model_path = tmp_path / "model.joblib"
    model_path.write_text("fake-model", encoding="utf-8")

    with ClearMLTracker(
        cfg,  # type: ignore[arg-type]
        stage="train:ridge",
        task_type="training",
        extra_tags=["ridge"],
    ) as tracker:
        tracker.connect_dict("params", {"alpha": 1.0})
        tracker.log_metrics({"val_rmse": 3.21}, series="ridge")
        tracker.upload_artifact("metrics", {"val_rmse": 3.21})
        model_id = tracker.register_model(
            model_path=model_path,
            model_name="housing-ridge",
            metadata={"val_rmse": 3.21},
        )

    assert model_id is not None
    assert len(_FakeTask.created) == 1
    task = _FakeTask.created[0]
    assert task.closed
    assert "ridge" in task.tags
    assert task.connected["params"]["alpha"] == 1.0
    assert any(row["title"] == "val_rmse" for row in task.logger.scalars)
    assert len(_FakeOutputModel.created) == 1
    assert _FakeOutputModel.created[0].weights_args["weights_filename"] == str(
        model_path
    )


def test_compare_clearml_experiments_returns_sorted_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    _install_fake_clearml(monkeypatch)

    t1 = _FakeTask({"task_name": "run-1"})
    t1.id = "task-1"
    t1.scalar_summary = {"val_rmse": {"main": {"last": 2.0}}}
    t2 = _FakeTask({"task_name": "run-2"})
    t2.id = "task-2"
    t2.scalar_summary = {"val_rmse": {"main": {"last": 1.0}}}
    _FakeTask.listed = [t1, t2]

    df = compare_clearml_experiments("boston-housing-hw5")
    assert isinstance(df, pd.DataFrame)
    assert list(df["task_id"]) == ["task-2", "task-1"]
    assert _FakeTask.last_project == "boston-housing-hw5"


def test_compare_registered_models_returns_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    _install_fake_clearml(monkeypatch)

    _FakeModel.listed = [
        _FakeRegistryModel(
            model_id="m1",
            name="housing-model",
            uri="file:///tmp/m1",
            framework="sklearn",
            published="2026-02-24T10:00:00+00:00",
            task="task-1",
            tags=["best"],
        )
    ]

    df = compare_registered_models(
        model_name="housing-model",
        project_name="boston-housing-hw5",
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.iloc[0]["model_id"] == "m1"
    assert _FakeModel.queried_with["project_name"] == "boston-housing-hw5"
