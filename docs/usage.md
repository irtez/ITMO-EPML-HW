# Примеры использования

## Конфигурация через Hydra

Проект использует [Hydra](https://hydra.cc/) для управления конфигурацией.

### Выбор алгоритма

```bash
# Линейная регрессия (по умолчанию)
poetry run python scripts/train.py

# Ridge
poetry run python scripts/train.py model=ridge

# Random Forest
poetry run python scripts/train.py model=random_forest
```

### CLI-оверрайды

```bash
# Изменить alpha у Ridge
poetry run python scripts/train.py model=ridge model.params.alpha=0.5

# Изменить параметры Random Forest
poetry run python scripts/train.py model=random_forest \
    model.params.n_estimators=200 \
    model.params.max_depth=15

# Изменить test_size
poetry run python scripts/train.py prepare.test_size=0.3

# Включить ClearML
poetry run python scripts/train.py model=random_forest clearml.enabled=true
```

### Доступные параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `model` | `linear_regression` | Алгоритм (linear_regression/ridge/random_forest) |
| `prepare.test_size` | `0.2` | Доля тестовой выборки |
| `prepare.random_state` | `42` | Сид для воспроизводимости |
| `train.val_size` | `0.2` | Доля валидационной выборки |
| `model.params.alpha` | `1.0` | Alpha для Ridge |
| `model.params.n_estimators` | `100` | Деревьев в RF |
| `model.params.max_depth` | `10` | Глубина деревьев RF |
| `clearml.enabled` | `false` | Включить ClearML трекинг |

---

## DVC-пайплайн

```bash
# Полный прогон (только изменённые стадии)
poetry run dvc repro

# Принудительный полный прогон
poetry run dvc repro --force

# Запустить конкретную стадию
poetry run dvc repro select_best

# Показать DAG пайплайна
poetry run dvc dag

# Показать метрики
poetry run dvc metrics show

# Показать diff метрик с предыдущим коммитом
poetry run dvc metrics diff
```

### Параллельное обучение (DVC queue)

```bash
poetry run dvc exp run --queue train_linear_regression
poetry run dvc exp run --queue train_ridge
poetry run dvc exp run --queue train_random_forest
poetry run dvc queue start -j 3
```

---

## MLflow

```bash
# Запуск 21 эксперимента (сетка гиперпараметров)
poetry run python scripts/run_experiments.py

# MLflow UI
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Python API трекинга

```python
from housing.tracking import ExperimentTracker

with ExperimentTracker(experiment_name="my-exp", run_name="run-1") as tracker:
    tracker.log_params({"alpha": 0.5, "model": "ridge"})
    tracker.log_metrics({"rmse": 4.2, "r2": 0.71})
    tracker.log_model(trained_model, artifact_path="model")
```

```python
from housing.tracking import mlflow_run

@mlflow_run(experiment_name="my-exp", run_name="experiment-1")
def train_model(X_train, y_train):
    model = Ridge(alpha=0.5)
    model.fit(X_train, y_train)
    return model
```

```python
from housing.tracking.utils import compare_runs, get_best_run

# Сравнить все прогоны эксперимента
df = compare_runs("boston-housing")
print(df[["run_name", "val_rmse", "val_r2"]].sort_values("val_rmse"))

# Получить лучший прогон
best_run_id = get_best_run("boston-housing", metric_name="val_rmse", mode="min")
```

---

## ClearML

```bash
# Запустить с ClearML трекингом
poetry run python scripts/train.py model=random_forest clearml.enabled=true

# Запустить ClearML Pipeline Controller
poetry run python scripts/clearml_pipeline.py clearml.enabled=true

# Экспортировать сравнение экспериментов и моделей
poetry run python scripts/clearml_compare.py clearml.enabled=true
# Результаты: metrics/clearml_experiments.csv, metrics/clearml_models.csv
```

---

## Генерация отчётов

```bash
# Сгенерировать отчёт с графиками (создаёт reports/report_hw6.md + figures)
poetry run python scripts/generate_report.py

# Проверить pipeline events
cat metrics/pipeline_events.jsonl | python -m json.tool | head -40
```

---

## Python API (прямое использование)

```python
from housing.data.load import load_raw_housing
from housing.features.build import get_feature_pipeline, split_features_target
from housing.models.train import compute_metrics, train_and_log_model

# Загрузить данные
df = load_raw_housing()

# Разделить на признаки и таргет
X, y = split_features_target(df)

# Получить пайплайн признаков
feature_pipeline = get_feature_pipeline()

# Обучить модель с MLflow трекингом
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
run_id, val_rmse = train_and_log_model(
    model_name="ridge",
    model=model,
    feature_pipeline=feature_pipeline,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    extra_params={"alpha": 1.0},
)
print(f"Run ID: {run_id}, Val RMSE: {val_rmse:.4f}")
```
