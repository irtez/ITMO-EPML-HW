"""Simple stage monitoring and notification helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from omegaconf import DictConfig


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def notify(cfg: DictConfig, message: str) -> None:
    """Send stage notification to configured sinks."""
    if bool(cfg.monitoring.notifications.console):
        print(message)

    if bool(cfg.monitoring.notifications.file):
        file_path = Path(str(cfg.monitoring.notifications.file_path))
        _append_line(file_path, message)


@contextmanager
def monitored_stage(cfg: DictConfig, stage: str) -> Iterator[None]:
    """Track stage lifecycle and emit start/success/failure notifications."""
    started_at = datetime.now(timezone.utc).isoformat()
    timer_start = perf_counter()
    notify(cfg, f"[{stage}] started at {started_at}")

    status = "success"
    error: str | None = None

    try:
        yield
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        finished_at = datetime.now(timezone.utc).isoformat()
        duration_sec = perf_counter() - timer_start

        event = {
            "stage": stage,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": round(duration_sec, 4),
        }
        if error is not None:
            event["error"] = error

        events_path = Path(str(cfg.monitoring.events_path))
        _append_line(events_path, json.dumps(event, ensure_ascii=True))

        notify(
            cfg,
            f"[{stage}] {status} at {finished_at} (duration={duration_sec:.2f}s)",
        )
