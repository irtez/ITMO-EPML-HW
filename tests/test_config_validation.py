"""Tests for Hydra config composition and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from housing.config import (
    validate_clearml_cfg,
    validate_evaluate_cfg,
    validate_prepare_cfg,
    validate_select_cfg,
    validate_train_cfg,
)


def _compose(*overrides: str) -> object:
    conf_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        return compose(config_name="pipeline", overrides=list(overrides))


@pytest.fixture
def composed_cfg() -> object:
    return _compose()


def test_hydra_default_model_is_linear_regression(composed_cfg: object) -> None:
    cfg = composed_cfg
    assert str(cfg.model.name) == "linear_regression"  # type: ignore[attr-defined]


def test_hydra_allows_model_override() -> None:
    cfg = _compose("model=random_forest")
    assert str(cfg.model.kind) == "random_forest"  # type: ignore[attr-defined]


def test_validate_prepare_cfg_accepts_defaults(composed_cfg: object) -> None:
    validate_prepare_cfg(composed_cfg)  # type: ignore[arg-type]


def test_validate_clearml_cfg_accepts_disabled(composed_cfg: object) -> None:
    validate_clearml_cfg(composed_cfg)  # type: ignore[arg-type]


def test_validate_clearml_cfg_requires_project_name_when_enabled() -> None:
    cfg = _compose("clearml.enabled=true", "clearml.project_name=' '")
    with pytest.raises(ValueError, match="clearml.project_name"):
        validate_clearml_cfg(cfg)


def test_validate_clearml_cfg_requires_api_host_format() -> None:
    cfg = _compose("clearml.enabled=true", "clearml.server.api_host=localhost:8008")
    with pytest.raises(ValueError, match="clearml.server.api_host"):
        validate_clearml_cfg(cfg)


def test_validate_train_cfg_rejects_invalid_ratio() -> None:
    cfg = _compose("train.val_size=1.2")

    with pytest.raises(ValueError, match="train.val_size"):
        validate_train_cfg(cfg)


def test_validate_select_cfg_requires_candidates() -> None:
    cfg = _compose("selection.candidates=[]")

    with pytest.raises(ValueError, match="selection.candidates"):
        validate_select_cfg(cfg)


def test_validate_prepare_cfg_rejects_invalid_ratio() -> None:
    cfg = _compose("prepare.test_size=0.0")
    with pytest.raises(ValueError, match="prepare.test_size"):
        validate_prepare_cfg(cfg)


def test_validate_prepare_cfg_rejects_negative_random_state() -> None:
    cfg = _compose("prepare.random_state=-1")
    with pytest.raises(ValueError, match="prepare.random_state"):
        validate_prepare_cfg(cfg)


def test_validate_train_cfg_rejects_unknown_model_kind() -> None:
    cfg = _compose("model.kind=svm")
    with pytest.raises(ValueError, match="model.kind"):
        validate_train_cfg(cfg)


def test_validate_train_cfg_rejects_negative_random_state() -> None:
    cfg = _compose("train.random_state=-1")
    with pytest.raises(ValueError, match="train.random_state"):
        validate_train_cfg(cfg)


def test_validate_train_cfg_accepts_ridge_defaults() -> None:
    cfg = _compose("model=ridge")
    validate_train_cfg(cfg)


def test_validate_train_cfg_rejects_non_positive_ridge_alpha() -> None:
    cfg = _compose("model=ridge", "model.params.alpha=0.0")
    with pytest.raises(ValueError, match="model.params.alpha"):
        validate_train_cfg(cfg)


def test_validate_train_cfg_accepts_random_forest_defaults() -> None:
    cfg = _compose("model=random_forest")
    validate_train_cfg(cfg)


def test_validate_train_cfg_rejects_random_forest_n_estimators() -> None:
    cfg = _compose("model=random_forest", "model.params.n_estimators=0")
    with pytest.raises(ValueError, match="n_estimators"):
        validate_train_cfg(cfg)


def test_validate_train_cfg_rejects_random_forest_max_depth() -> None:
    cfg = _compose("model=random_forest", "model.params.max_depth=0")
    with pytest.raises(ValueError, match="max_depth"):
        validate_train_cfg(cfg)


def test_validate_select_cfg_rejects_empty_name() -> None:
    cfg = _compose()
    cfg.selection.candidates[0].name = ""  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="candidates\\[0\\]\\.name"):
        validate_select_cfg(cfg)


def test_validate_select_cfg_rejects_empty_model_path() -> None:
    cfg = _compose()
    cfg.selection.candidates[0].model_path = ""  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="candidates\\[0\\]\\.model_path"):
        validate_select_cfg(cfg)


def test_validate_select_cfg_rejects_empty_metrics_path() -> None:
    cfg = _compose()
    cfg.selection.candidates[0].metrics_path = ""  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="candidates\\[0\\]\\.metrics_path"):
        validate_select_cfg(cfg)


def test_validate_select_cfg_accepts_defaults(composed_cfg: object) -> None:
    validate_select_cfg(composed_cfg)  # type: ignore[arg-type]


def test_validate_evaluate_cfg_accepts_defaults(composed_cfg: object) -> None:
    validate_evaluate_cfg(composed_cfg)  # type: ignore[arg-type]


def test_validate_evaluate_cfg_rejects_empty_model_path() -> None:
    cfg = _compose("evaluate.input.model_path=' '")
    with pytest.raises(ValueError, match="evaluate.input.model_path"):
        validate_evaluate_cfg(cfg)


def test_validate_evaluate_cfg_rejects_empty_test_path() -> None:
    cfg = _compose("evaluate.input.test_path=' '")
    with pytest.raises(ValueError, match="evaluate.input.test_path"):
        validate_evaluate_cfg(cfg)


def test_validate_evaluate_cfg_rejects_empty_metrics_path() -> None:
    cfg = _compose("evaluate.output.metrics_path=' '")
    with pytest.raises(ValueError, match="evaluate.output.metrics_path"):
        validate_evaluate_cfg(cfg)
