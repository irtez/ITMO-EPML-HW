"""Tests for pipeline monitoring and notifications."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from housing.pipeline.monitoring import monitored_stage, notify


def _monitoring_cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "monitoring": {
                "events_path": str(tmp_path / "events.jsonl"),
                "notifications": {
                    "console": False,
                    "file": True,
                    "file_path": str(tmp_path / "notify.log"),
                },
            }
        }
    )


def test_notify_writes_file_message(tmp_path: Path) -> None:
    cfg = _monitoring_cfg(tmp_path)
    notify(cfg, "hello")  # type: ignore[arg-type]

    log_path = tmp_path / "notify.log"
    assert log_path.exists()
    assert "hello" in log_path.read_text(encoding="utf-8")


def test_notify_prints_to_console_when_enabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = OmegaConf.create(
        {
            "monitoring": {
                "events_path": str(tmp_path / "events.jsonl"),
                "notifications": {
                    "console": True,
                    "file": False,
                    "file_path": str(tmp_path / "notify.log"),
                },
            }
        }
    )
    notify(cfg, "stdout-message")  # type: ignore[arg-type]
    captured = capsys.readouterr()
    assert "stdout-message" in captured.out


def test_monitored_stage_writes_success_event(tmp_path: Path) -> None:
    cfg = _monitoring_cfg(tmp_path)

    with monitored_stage(cfg, "stage-a"):
        pass

    events_path = tmp_path / "events.jsonl"
    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["stage"] == "stage-a"
    assert rows[0]["status"] == "success"


def test_monitored_stage_writes_failure_event(tmp_path: Path) -> None:
    cfg = _monitoring_cfg(tmp_path)

    try:
        with monitored_stage(cfg, "stage-b"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["stage"] == "stage-b"
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["error"] == "boom"
