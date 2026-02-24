"""Automatic experiment report generator for ITMO EPML HW.

Reads DVC pipeline metrics from ``metrics/`` and produces:

  reports/experiments/
  ├── figures/
  │   ├── model_comparison.png   – val RMSE / MAE / R² bar charts
  │   ├── train_val_gap.png      – train vs val RMSE (overfitting view)
  │   └── test_metrics.png       – best-model test-set metrics
  └── report_YYYY-MM-DD_HH-MM-SS.md  – timestamped Markdown report

Usage::

    poetry run python scripts/generate_report.py
    poetry run python scripts/generate_report.py --out-dir reports/experiments
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "metrics"
CONF_MODEL_DIR = ROOT / "conf" / "model"


def _load_model_configs() -> dict[str, dict]:  # type: ignore[type-arg]
    """Read per-model Hydra configs from conf/model/*.yaml."""
    configs: dict[str, dict] = {}  # type: ignore[type-arg]
    for path in CONF_MODEL_DIR.glob("*.yaml"):
        with path.open() as fh:
            cfg = yaml.safe_load(fh)
        name = cfg.get("name", path.stem)
        configs[name] = cfg
    return configs


def _params_str(params: dict) -> str:  # type: ignore[type-arg]
    """Format params dict as readable string, or '-' if empty."""
    if not params:
        return "-"
    return ", ".join(f"{k} = {v}" for k, v in params.items())


def _load_pipeline_cfg() -> dict:  # type: ignore[type-arg]
    """Read prepare/train settings from conf/pipeline.yaml."""
    with (ROOT / "conf" / "pipeline.yaml").open() as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


def _gap_summary(models_data: dict) -> str:  # type: ignore[type-arg]
    """Data-driven bullet list: gap level per model (no fixed model names)."""
    lines = []
    for m in MODEL_NAMES:
        gap = models_data[m]["val_rmse"] - models_data[m]["train_rmse"]
        level = (
            "минимальное"
            if gap < 0.5
            else ("умеренное" if gap < 2.5 else "значительное")
        )
        lines.append(
            f"- **{MODEL_LABELS[m]}**: gap = {_fmt(gap)} → {level} переобучение"
        )
    return "\n".join(lines)


def _auto_conclusions(
    md: dict,  # type: ignore[type-arg]
    test_metrics: dict,  # type: ignore[type-arg]
    best: str,
    best_val_rmse: float,
) -> str:
    """Fully data-driven conclusion bullets."""
    worst = max(MODEL_NAMES, key=lambda m: md[m]["val_rmse"])
    improvement_pct = (
        (md[worst]["val_rmse"] - best_val_rmse) / md[worst]["val_rmse"] * 100
    )

    min_gap_m = min(MODEL_NAMES, key=lambda m: md[m]["val_rmse"] - md[m]["train_rmse"])
    max_gap_m = max(MODEL_NAMES, key=lambda m: md[m]["val_rmse"] - md[m]["train_rmse"])
    min_gap = md[min_gap_m]["val_rmse"] - md[min_gap_m]["train_rmse"]
    max_gap = md[max_gap_m]["val_rmse"] - md[max_gap_m]["train_rmse"]

    return "\n".join(
        [
            f"1. **Лучшая модель:** {MODEL_LABELS[best]}"
            f" - val RMSE = {_fmt(best_val_rmse)}",
            f"2. **Снижение val RMSE** относительно худшей"
            f" ({MODEL_LABELS[worst]}): **{improvement_pct:.1f}%**",
            f"3. **Наименьший gap** train/val: {MODEL_LABELS[min_gap_m]}"
            f" (gap = {_fmt(min_gap)})",
            f"4. **Наибольший gap** train/val: {MODEL_LABELS[max_gap_m]}"
            f" (gap = {_fmt(max_gap)})",
            f"5. **Тестовые метрики** ({MODEL_LABELS[best]}): "
            f"RMSE = {_fmt(test_metrics['rmse'])}, "
            f"MAE = {_fmt(test_metrics['mae'])}, "
            f"R² = {_fmt(test_metrics['r2'])}",
        ]
    )


MODEL_NAMES = ["linear_regression", "ridge", "random_forest"]
MODEL_LABELS: dict[str, str] = {
    "linear_regression": "Linear Regression",
    "ridge": "Ridge (α=1.0)",
    "random_forest": "Random Forest",
}

# colour palette
C_BLUE = "#4361ee"
C_TEAL = "#06d6a0"
C_RED = "#ef476f"
C_YELLOW = "#ffd166"


# ── helpers ───────────────────────────────────────────────────────────────────


def _load(path: Path) -> dict:  # type: ignore[type-arg]
    with path.open() as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


# ── plots ─────────────────────────────────────────────────────────────────────


def plot_model_comparison(
    models_data: dict,  # type: ignore[type-arg]
    best: str,
    fig_dir: Path,
) -> Path:
    """Three side-by-side bar charts: val RMSE / MAE / R²."""
    labels = [MODEL_LABELS[m] for m in MODEL_NAMES]
    metrics = {
        "Val RMSE\n(↓ лучше)": [models_data[m]["val_rmse"] for m in MODEL_NAMES],
        "Val MAE\n(↓ лучше)": [models_data[m]["val_mae"] for m in MODEL_NAMES],
        "Val R²\n(↑ лучше)": [models_data[m]["val_r2"] for m in MODEL_NAMES],
    }

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Сравнение моделей - валидационная выборка", fontsize=13, y=1.01)

    colours = [C_BLUE, C_TEAL, C_RED]
    for ax, (title, vals), clr in zip(axes, metrics.items(), colours, strict=False):
        bar_colours = [
            C_YELLOW if MODEL_NAMES[i] == best else clr for i in range(len(labels))
        ]
        bars = ax.bar(
            x, vals, color=bar_colours, alpha=0.9, edgecolor="white", width=0.5
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=13, ha="right", fontsize=9)
        ax.bar_label(bars, fmt="{:.3f}", padding=3, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.28)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        -0.03,
        f"Жёлтый = лучшая модель ({MODEL_LABELS[best]})",
        ha="center",
        fontsize=9,
        color="#555",
    )
    plt.tight_layout()
    out = fig_dir / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved %s", out.relative_to(ROOT))
    return out


def plot_train_val_gap(
    models_data: dict,  # type: ignore[type-arg]
    fig_dir: Path,
) -> Path:
    """Grouped bar chart: train RMSE vs val RMSE per model."""
    labels = [MODEL_LABELS[m] for m in MODEL_NAMES]
    train_rmse = [models_data[m]["train_rmse"] for m in MODEL_NAMES]
    val_rmse = [models_data[m]["val_rmse"] for m in MODEL_NAMES]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, train_rmse, w, label="Train RMSE", color=C_BLUE, alpha=0.87)
    b2 = ax.bar(x + w / 2, val_rmse, w, label="Val RMSE", color=C_RED, alpha=0.87)

    ax.set_title("Train vs Validation RMSE - анализ переобучения", fontsize=12)
    ax.set_ylabel("RMSE (тыс. $)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.bar_label(b1, fmt="{:.2f}", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="{:.2f}", padding=2, fontsize=9)
    ax.set_ylim(0, max(val_rmse) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = fig_dir / "train_val_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved %s", out.relative_to(ROOT))
    return out


def plot_test_metrics(test_metrics: dict, best: str, fig_dir: Path) -> Path:  # type: ignore[type-arg]
    """Horizontal bar chart for test-set metrics of the best model."""
    names = ["RMSE (тыс. $)", "MAE (тыс. $)", "R²"]
    values = [test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"]]
    colours = [C_RED, C_YELLOW, C_TEAL]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(
        names, values, color=colours, alpha=0.9, edgecolor="white", height=0.45
    )
    ax.set_title(
        f"Тестовые метрики - {MODEL_LABELS[best]} (лучшая модель)", fontsize=11
    )
    ax.bar_label(bars, fmt="{:.4f}", padding=5, fontsize=11)
    ax.set_xlim(0, max(values) * 1.35)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = fig_dir / "test_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved %s", out.relative_to(ROOT))
    return out


# ── markdown report ───────────────────────────────────────────────────────────


def _comparison_table(models_data: dict, best: str) -> str:  # type: ignore[type-arg]
    header = (
        "| Модель | Train RMSE | Val RMSE | Val MAE | Val R² | Итог |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = ""
    for m in MODEL_NAMES:
        d = models_data[m]
        flag = "✅ **Лучшая**" if m == best else "-"
        rows += (
            f"| {MODEL_LABELS[m]} "
            f"| {_fmt(d['train_rmse'])} "
            f"| {_fmt(d['val_rmse'])} "
            f"| {_fmt(d['val_mae'])} "
            f"| {_fmt(d['val_r2'])} "
            f"| {flag} |\n"
        )
    return header + rows


def _gap_table(models_data: dict) -> str:  # type: ignore[type-arg]
    header = (
        "| Модель | Train RMSE | Val RMSE | Gap (Val − Train) | Оценка |\n"
        "|---|---|---|---|---|\n"
    )
    rows = ""
    for m in MODEL_NAMES:
        d = models_data[m]
        gap = d["val_rmse"] - d["train_rmse"]
        verdict = (
            "Минимальное" if gap < 0.5 else ("Умеренное" if gap < 2.5 else "Сильное")
        )
        rows += (
            f"| {MODEL_LABELS[m]} "
            f"| {_fmt(d['train_rmse'])} "
            f"| {_fmt(d['val_rmse'])} "
            f"| **{_fmt(gap)}** "
            f"| {verdict} |\n"
        )
    return header + rows


def build_report(
    train_metrics: dict,  # type: ignore[type-arg]
    test_metrics: dict,  # type: ignore[type-arg]
    per_model: dict,  # type: ignore[type-arg]
    model_configs: dict,  # type: ignore[type-arg]
    pipeline_cfg: dict,  # type: ignore[type-arg]
    report_ts: datetime,
) -> str:
    best = train_metrics["best_model"]
    best_label = MODEL_LABELS[best]
    md = train_metrics["all_models"]

    # Params table from actual Hydra model configs
    params_rows = ""
    for m in MODEL_NAMES:
        cfg = model_configs.get(m, {})
        params = cfg.get("params", {})
        params_rows += f"| {MODEL_LABELS[m]} | {_params_str(params)} |\n"

    # Pipeline config values (read from conf/pipeline.yaml)
    test_size = pipeline_cfg.get("prepare", {}).get("test_size", "-")
    val_size = pipeline_cfg.get("train", {}).get("val_size", "-")
    random_state = pipeline_cfg.get("prepare", {}).get("random_state", "-")
    selection_metric = "Val RMSE (минимизация)"

    return f"""# Отчёт об экспериментах - {report_ts.strftime("%d.%m.%Y %H:%M:%S")}

> Автоматически сгенерировано скриптом `scripts/generate_report.py`
> Данные: `metrics/` (выходы DVC-пайплайна)

---

## 1. Конфигурация пайплайна

| Параметр | Значение |
|---|---|
| Test size | {test_size} |
| Val size | {val_size} |
| Random state | {random_state} |
| Метрика выбора | {selection_metric} |

> Полная конфигурация: `conf/pipeline.yaml`, `conf/model/*.yaml`

---

## 2. Результаты обучения

### 2.1 Сравнительная таблица моделей

{_comparison_table(md, best)}

![Сравнение моделей](figures/model_comparison.png)

### 2.2 Параметры моделей

| Модель | Гиперпараметры |
|---|---|
{params_rows}
> Конфигурация: `conf/model/*.yaml`

---

## 3. Анализ переобучения (train vs val RMSE)

{_gap_table(md)}

{_gap_summary(md)}

![Train vs Val RMSE](figures/train_val_gap.png)

---

## 4. Результаты на тестовой выборке

Лучшая модель по валидации - **{best_label}** - оценена на отложенной тестовой выборке:

| Метрика | Значение |
|---|---|
| **RMSE** | **{_fmt(test_metrics["rmse"])}** |
| **MAE** | **{_fmt(test_metrics["mae"])}** |
| **R²** | **{_fmt(test_metrics["r2"])}** |

![Тестовые метрики](figures/test_metrics.png)

---

## 5. Итоговые показатели

{_auto_conclusions(md, test_metrics, best, train_metrics["best_val_rmse"])}

---

## 6. Воспроизведение результатов

```bash
# Полный прогон пайплайна
poetry run dvc repro

# Пересоздать этот отчёт
poetry run python scripts/generate_report.py
```

Параметры зафиксированы в `conf/pipeline.yaml` и `conf/model/*.yaml`.
Данные версионированы через DVC (`dvc.lock`).
Зависимости зафиксированы в `poetry.lock`.
"""


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument(
        "--out-dir",
        default="reports/experiments",
        help="Output directory (default: reports/experiments)",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading metrics from %s", METRICS_DIR.relative_to(ROOT))
    train_metrics = _load(METRICS_DIR / "train_metrics.json")
    test_metrics = _load(METRICS_DIR / "test_metrics.json")
    per_model: dict = {}  # type: ignore[type-arg]
    for m in MODEL_NAMES:
        p = METRICS_DIR / f"train_{m}.json"
        if p.exists():
            per_model[m] = _load(p)

    log.info("Loading model configs from %s", CONF_MODEL_DIR.relative_to(ROOT))
    model_configs = _load_model_configs()
    pipeline_cfg = _load_pipeline_cfg()

    best = train_metrics["best_model"]
    models_data = train_metrics["all_models"]

    log.info("Generating figures → %s", fig_dir.relative_to(ROOT))
    plot_model_comparison(models_data, best, fig_dir)
    plot_train_val_gap(models_data, fig_dir)
    plot_test_metrics(test_metrics, best, fig_dir)

    report_ts = datetime.now()
    report_text = build_report(
        train_metrics, test_metrics, per_model, model_configs, pipeline_cfg, report_ts
    )

    ts_str = report_ts.strftime("%Y-%m-%d_%H-%M-%S")
    report_path = out_dir / f"report_{ts_str}.md"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Report → %s", report_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
