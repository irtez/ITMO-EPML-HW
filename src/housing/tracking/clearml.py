"""ClearML integration helpers for pipeline stages and experiment analysis."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf


class ClearMLUnavailableError(RuntimeError):
    """Raised when ClearML is enabled but the runtime is not available."""


def _load_clearml() -> Any:
    """Import ``clearml`` lazily to keep disabled mode lightweight."""
    try:
        return importlib.import_module("clearml")
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise ClearMLUnavailableError(
            "ClearML is enabled but cannot be imported. "
            "Check installation and local runtime permissions."
        ) from exc


def _cfg_dict(cfg: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Hydra config must resolve to a dictionary")
    return cast(dict[str, Any], {str(key): value for key, value in raw.items()})


def _resolve_task_type(task_cls: Any, task_type: str) -> Any:
    task_types = getattr(task_cls, "TaskTypes", None)
    if task_types is None:
        return task_type

    for candidate in (task_type, task_type.lower(), task_type.upper()):
        if hasattr(task_types, candidate):
            return getattr(task_types, candidate)

    if hasattr(task_types, "training"):
        return task_types.training
    return task_type


class ClearMLTracker:
    """Context manager around a single ClearML task."""

    def __init__(
        self,
        cfg: DictConfig,
        stage: str,
        task_type: str = "training",
        extra_tags: list[str] | None = None,
    ) -> None:
        self._cfg = cfg
        self._stage = stage
        self._task_type = task_type
        self._extra_tags = extra_tags or []
        self._task: Any = None
        self._logger: Any = None

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.clearml.enabled)

    @property
    def task_id(self) -> str | None:
        if self._task is None:
            return None
        task_id = getattr(self._task, "id", None)
        return str(task_id) if task_id is not None else None

    def __enter__(self) -> ClearMLTracker:
        if not self.enabled:
            return self

        clearml = _load_clearml()
        task_cls = clearml.Task
        task_name = f"{self._cfg.clearml.task_prefix}:{self._stage}"
        kwargs: dict[str, Any] = {
            "project_name": str(self._cfg.clearml.project_name),
            "task_name": task_name,
            "task_type": _resolve_task_type(task_cls, self._task_type),
        }

        output_uri = str(self._cfg.clearml.output_uri).strip()
        if output_uri:
            kwargs["output_uri"] = output_uri

        self._task = task_cls.init(**kwargs)
        tags = [str(tag) for tag in list(self._cfg.clearml.tags)] + self._extra_tags
        if tags:
            self._task.add_tags(tags)

        self._task.connect({"stage": self._stage}, name="runtime")
        self._task.connect_configuration(
            name="pipeline_cfg",
            configuration=_cfg_dict(self._cfg),
        )
        self._logger = self._task.get_logger()
        self.log_text(f"[{self._stage}] ClearML task started")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if not self.enabled or self._task is None:
            return

        if exc_type is None:
            self.log_text(f"[{self._stage}] finished successfully")
        else:
            self.log_text(f"[{self._stage}] failed: {exc_val}")
        self._task.close()

    def connect_dict(self, name: str, payload: dict[str, Any]) -> None:
        """Attach a parameter dictionary to the active task."""
        if self._task is None:
            return
        self._task.connect(payload, name=name)

    def log_metrics(
        self,
        metrics: dict[str, float],
        series: str = "main",
        iteration: int = 0,
    ) -> None:
        """Log a metric dictionary as scalar series."""
        if self._logger is None:
            return

        for metric_name, metric_value in metrics.items():
            self._logger.report_scalar(
                title=str(metric_name),
                series=series,
                value=float(metric_value),
                iteration=iteration,
            )

    def log_text(self, message: str) -> None:
        """Write a text log line to the task output."""
        if self._logger is None:
            return
        self._logger.report_text(message)

    def upload_artifact(self, name: str, artifact: Any) -> None:
        """Upload path/object as a task artifact."""
        if self._task is None:
            return

        artifact_object: Any = artifact
        if isinstance(artifact, Path):
            artifact_object = str(artifact)
        self._task.upload_artifact(name=name, artifact_object=artifact_object)

    def register_model(
        self,
        model_path: Path,
        model_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Register a model version in ClearML model registry."""
        if self._task is None:
            return None

        clearml = _load_clearml()
        output_model_cls = getattr(clearml, "OutputModel", None)
        if output_model_cls is None:
            self.upload_artifact(f"model::{model_name}", model_path)
            return None

        output_model = output_model_cls(task=self._task, name=model_name)
        output_model.update_weights(
            weights_filename=str(model_path),
            target_filename=model_path.name,
            auto_delete_file=False,
        )
        if metadata is not None:
            metadata_text = {str(key): str(value) for key, value in metadata.items()}
            output_model.update_design(config_dict={"model_metadata": metadata_text})
            output_model.comment = (
                f"Auto-registered by stage '{self._stage}' from {model_path.name}"
            )
            for key, value in metadata.items():
                output_model.set_metadata(
                    key=str(key),
                    value=str(value),
                    v_type=type(value).__name__,
                )
            self.upload_artifact(f"metadata::{model_name}", metadata)

        model_id = getattr(output_model, "id", None)
        return str(model_id) if model_id is not None else None


def _extract_last_scalar(
    scalar_summary: dict[str, Any],
    metric_name: str,
) -> float | None:
    for title, series_map in scalar_summary.items():
        if not isinstance(series_map, dict):
            continue
        for series_name, stats in series_map.items():
            if not isinstance(stats, dict):
                continue
            is_target = (
                title == metric_name
                or title.endswith(f"/{metric_name}")
                or series_name == metric_name
            )
            if not is_target:
                continue
            value = stats.get("last")
            if isinstance(value, float | int):
                return float(value)
    return None


def compare_clearml_experiments(
    project_name: str,
    metric_name: str = "val_rmse",
) -> pd.DataFrame:
    """Return a comparison table for all tasks in a ClearML project."""
    clearml = _load_clearml()
    task_cls = clearml.Task
    tasks = task_cls.get_tasks(project_name=project_name) or []

    rows: list[dict[str, Any]] = []
    for task in tasks:
        scalar_summary: dict[str, Any] = {}
        if hasattr(task, "get_last_scalar_metrics"):
            scalar_summary = task.get_last_scalar_metrics() or {}

        rows.append(
            {
                "task_id": str(getattr(task, "id", "")),
                "task_name": str(getattr(task, "name", "")),
                "status": str(getattr(task, "status", "")),
                metric_name: _extract_last_scalar(scalar_summary, metric_name),
                "started": str(getattr(task, "started", "")),
                "completed": str(getattr(task, "completed", "")),
                "tags": ",".join(getattr(task, "tags", []) or []),
            }
        )

    df = pd.DataFrame(rows)
    if metric_name in df.columns:
        df = df.sort_values(metric_name, ascending=True, na_position="last")
    return df.reset_index(drop=True)


def compare_registered_models(
    model_name: str,
    project_name: str | None = None,
    max_results: int = 100,
) -> pd.DataFrame:
    """Return a comparison table for model versions from ClearML registry."""
    clearml = _load_clearml()
    model_cls = clearml.Model

    query_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "max_results": max_results,
    }
    if project_name:
        query_kwargs["project_name"] = project_name

    models = model_cls.query_models(**query_kwargs) or []
    rows: list[dict[str, Any]] = []
    for model in models:
        rows.append(
            {
                "model_id": str(getattr(model, "id", "")),
                "name": str(getattr(model, "name", "")),
                "uri": str(getattr(model, "uri", "")),
                "framework": str(getattr(model, "framework", "")),
                "published": str(getattr(model, "published", "")),
                "task": str(getattr(model, "task", "")),
                "tags": ",".join(getattr(model, "tags", []) or []),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ClearMLTracker",
    "ClearMLUnavailableError",
    "compare_clearml_experiments",
    "compare_registered_models",
]
