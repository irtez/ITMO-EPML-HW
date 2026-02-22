"""Validation utilities for composed Hydra configurations."""

from __future__ import annotations

from omegaconf import DictConfig

_ALLOWED_MODEL_KINDS = {"linear_regression", "ridge", "random_forest"}


def _require_positive_float(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _require_ratio(value: float, name: str) -> None:
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")


def _require_non_empty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_http_url(value: str, name: str) -> None:
    _require_non_empty(value, name)
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError(f"{name} must start with http:// or https://, got {value}")


def validate_clearml_cfg(cfg: DictConfig) -> None:
    """Validate ClearML config values shared by all stages."""
    if not bool(cfg.clearml.enabled):
        return

    _require_non_empty(str(cfg.clearml.project_name), "clearml.project_name")
    _require_non_empty(str(cfg.clearml.task_prefix), "clearml.task_prefix")
    _require_non_empty(str(cfg.clearml.model_name), "clearml.model_name")
    _require_non_empty(
        str(cfg.clearml.pipeline.execution_queue),
        "clearml.pipeline.execution_queue",
    )
    _require_non_empty(str(cfg.clearml.pipeline.name), "clearml.pipeline.name")
    _require_non_empty(
        str(cfg.clearml.auth.access_key_env),
        "clearml.auth.access_key_env",
    )
    _require_non_empty(
        str(cfg.clearml.auth.secret_key_env),
        "clearml.auth.secret_key_env",
    )
    _require_http_url(str(cfg.clearml.server.api_host), "clearml.server.api_host")
    _require_http_url(str(cfg.clearml.server.web_host), "clearml.server.web_host")
    _require_http_url(str(cfg.clearml.server.files_host), "clearml.server.files_host")


def validate_prepare_cfg(cfg: DictConfig) -> None:
    """Validate config values used in prepare stage."""
    validate_clearml_cfg(cfg)
    _require_ratio(float(cfg.prepare.test_size), "prepare.test_size")
    if int(cfg.prepare.random_state) < 0:
        raise ValueError("prepare.random_state must be >= 0")


def validate_train_cfg(cfg: DictConfig) -> None:
    """Validate config values used in train stage."""
    validate_clearml_cfg(cfg)
    model_kind = str(cfg.model.kind)
    if model_kind not in _ALLOWED_MODEL_KINDS:
        raise ValueError(
            f"model.kind must be one of {_ALLOWED_MODEL_KINDS}, got {model_kind}"
        )

    _require_ratio(float(cfg.train.val_size), "train.val_size")
    if int(cfg.train.random_state) < 0:
        raise ValueError("train.random_state must be >= 0")

    if model_kind == "ridge":
        alpha = float(cfg.model.params.alpha)
        _require_positive_float(alpha, "model.params.alpha")

    if model_kind == "random_forest":
        n_estimators = int(cfg.model.params.n_estimators)
        if n_estimators < 1:
            raise ValueError("model.params.n_estimators must be >= 1")

        max_depth = int(cfg.model.params.max_depth)
        if max_depth < 1:
            raise ValueError("model.params.max_depth must be >= 1")


def validate_select_cfg(cfg: DictConfig) -> None:
    """Validate config values used in best-model selection stage."""
    validate_clearml_cfg(cfg)
    if len(cfg.selection.candidates) == 0:
        raise ValueError("selection.candidates must contain at least one candidate")

    for idx, item in enumerate(cfg.selection.candidates):
        if not str(item.name).strip():
            raise ValueError(f"selection.candidates[{idx}].name must be non-empty")
        if not str(item.model_path).strip():
            raise ValueError(
                f"selection.candidates[{idx}].model_path must be non-empty"
            )
        if not str(item.metrics_path).strip():
            raise ValueError(
                f"selection.candidates[{idx}].metrics_path must be non-empty"
            )


def validate_evaluate_cfg(cfg: DictConfig) -> None:
    """Validate config values used in evaluation stage."""
    validate_clearml_cfg(cfg)
    if not str(cfg.evaluate.input.model_path).strip():
        raise ValueError("evaluate.input.model_path must be non-empty")
    if not str(cfg.evaluate.input.test_path).strip():
        raise ValueError("evaluate.input.test_path must be non-empty")
    if not str(cfg.evaluate.output.metrics_path).strip():
        raise ValueError("evaluate.output.metrics_path must be non-empty")
