"""Build and run the full ML workflow as a ClearML pipeline."""

import shlex
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from housing.config import validate_clearml_cfg


def _run_shell_stage(command: str) -> str:
    """Run one local shell command for a pipeline step."""
    args = shlex.split(command)
    completed = subprocess.run(  # noqa: S603  # nosec B603
        args,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _build_pipeline(cfg: DictConfig) -> Any:
    import importlib

    clearml = importlib.import_module("clearml")
    pipeline_cls = clearml.PipelineController

    pipeline = pipeline_cls(
        project=str(cfg.clearml.project_name),
        name=str(cfg.clearml.pipeline.name),
        version="1.0.0",
    )
    pipeline.set_default_execution_queue(str(cfg.clearml.pipeline.execution_queue))

    pipeline.add_function_step(
        name="prepare",
        function=_run_shell_stage,
        function_kwargs={"command": "python scripts/prepare.py clearml.enabled=true"},
        function_return=["stdout"],
        cache_executed_step=False,
    )
    pipeline.add_function_step(
        name="train_linear_regression",
        parents=["prepare"],
        function=_run_shell_stage,
        function_kwargs={
            "command": (
                "python scripts/train.py model=linear_regression clearml.enabled=true"
            )
        },
        function_return=["stdout"],
        cache_executed_step=False,
    )
    pipeline.add_function_step(
        name="train_ridge",
        parents=["prepare"],
        function=_run_shell_stage,
        function_kwargs={
            "command": "python scripts/train.py model=ridge clearml.enabled=true"
        },
        function_return=["stdout"],
        cache_executed_step=False,
    )
    pipeline.add_function_step(
        name="train_random_forest",
        parents=["prepare"],
        function=_run_shell_stage,
        function_kwargs={
            "command": (
                "python scripts/train.py model=random_forest clearml.enabled=true"
            )
        },
        function_return=["stdout"],
        cache_executed_step=False,
    )
    pipeline.add_function_step(
        name="select_best",
        parents=["train_linear_regression", "train_ridge", "train_random_forest"],
        function=_run_shell_stage,
        function_kwargs={
            "command": "python scripts/select_best.py clearml.enabled=true"
        },
        function_return=["stdout"],
        cache_executed_step=False,
    )
    pipeline.add_function_step(
        name="evaluate",
        parents=["select_best"],
        function=_run_shell_stage,
        function_kwargs={"command": "python scripts/evaluate.py clearml.enabled=true"},
        function_return=["stdout"],
        cache_executed_step=False,
    )

    return pipeline


def _set_pipeline_metadata(pipeline: Any) -> None:
    """Set description/tags with compatibility across ClearML versions."""
    description = (
        "DVC/Hydra workflow mirrored in ClearML: "
        "prepare -> train_* -> select -> evaluate"
    )
    tags = ["hw5", "pipeline", "boston-housing"]

    if hasattr(pipeline, "set_description"):
        pipeline.set_description(description)
    elif hasattr(pipeline, "_task") and hasattr(pipeline._task, "set_comment"):
        pipeline._task.set_comment(description)

    if hasattr(pipeline, "add_tags"):
        pipeline.add_tags(tags)
    elif hasattr(pipeline, "_task") and hasattr(pipeline._task, "add_tags"):
        pipeline._task.add_tags(tags)


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Create and run the ClearML pipeline controller."""
    validate_clearml_cfg(cfg)
    if not bool(cfg.clearml.enabled):
        print("ClearML disabled (clearml.enabled=false), pipeline was not started.")
        return

    pipeline = _build_pipeline(cfg)
    _set_pipeline_metadata(pipeline)

    if bool(cfg.clearml.pipeline.run_locally):
        pipeline.start_locally(run_pipeline_steps_locally=True)
    else:
        pipeline.start()
    pipeline.wait()

    output_note = Path("metrics") / "clearml_pipeline_schedule.txt"
    output_note.parent.mkdir(parents=True, exist_ok=True)
    output_note.write_text(
        (
            "Configured cron schedule for this pipeline:\n"
            f"{cfg.clearml.pipeline.schedule_cron}\n\n"
            "Apply this expression in system scheduler (cron) for automatic runs.\n"
        ),
        encoding="utf-8",
    )
    print("ClearML pipeline finished.")
    print(f"Schedule note saved: {output_note}")


if __name__ == "__main__":
    main()
