# housing.tracking - Трекинг экспериментов

Модуль: `src/housing/tracking/`

Предоставляет три паттерна интеграции с MLflow и один с ClearML.

---

## MLflow: ExperimentTracker (context manager)

Модуль: `src/housing/tracking/context.py`

```python
class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None
```

Контекстный менеджер для MLflow-запуска.

**Методы:**

| Метод | Описание |
|---|---|
| `log_param(key, value)` | Залогировать один параметр |
| `log_params(params_dict)` | Залогировать словарь параметров |
| `log_metric(key, value)` | Залогировать одну метрику |
| `log_metrics(metrics_dict)` | Залогировать словарь метрик |
| `set_tag(key, value)` | Установить один тег |
| `set_tags(tags_dict)` | Установить словарь тегов |
| `log_model(model, artifact_path)` | Сохранить модель как артефакт |
| `end_run()` | Завершить MLflow-запуск |

**Пример:**

```python
from housing.tracking import ExperimentTracker

with ExperimentTracker(
    experiment_name="boston-housing",
    run_name="ridge-alpha-0.5",
    tags={"stage": "training", "model": "ridge"},
) as tracker:
    tracker.log_params({"alpha": 0.5, "model_type": "ridge"})
    # ... обучение ...
    tracker.log_metrics({"val_rmse": 4.2, "val_r2": 0.71})
    tracker.log_model(trained_pipeline, artifact_path="model")
```

---

## MLflow: @mlflow_run (decorator)

Модуль: `src/housing/tracking/decorators.py`

```python
def mlflow_run(
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Callable
```

Декоратор, оборачивающий функцию в MLflow-запуск.

**Пример:**

```python
from housing.tracking import mlflow_run

@mlflow_run(
    experiment_name="boston-housing",
    run_name="ridge-experiment",
    tags={"model": "ridge"},
)
def train_ridge(X_train, y_train, X_val, y_val, alpha=1.0):
    import mlflow
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    val_rmse = compute_rmse(model, X_val, y_val)
    mlflow.log_metric("val_rmse", val_rmse)
    return model
```

---

## MLflow: утилиты

Модуль: `src/housing/tracking/utils.py`

### `compare_runs`

```python
def compare_runs(experiment_name: str) -> pd.DataFrame
```

Возвращает DataFrame со всеми прогонами эксперимента.

```python
from housing.tracking.utils import compare_runs

df = compare_runs("boston-housing")
print(df[["run_name", "val_rmse", "val_r2"]].sort_values("val_rmse"))
```

### `get_best_run`

```python
def get_best_run(
    experiment_name: str,
    metric_name: str,
    mode: str = "min",
) -> str
```

Возвращает `run_id` лучшего прогона по указанной метрике.

| Параметр | Тип | Описание |
|---|---|---|
| `experiment_name` | `str` | Название эксперимента |
| `metric_name` | `str` | Метрика для сравнения (например, `val_rmse`) |
| `mode` | `str` | `"min"` или `"max"` |

```python
from housing.tracking.utils import get_best_run

best_id = get_best_run("boston-housing", metric_name="val_rmse", mode="min")
print(f"Best run: {best_id}")
```

### `search_runs`

```python
def search_runs(
    experiment_name: str,
    filter_string: str = "",
    max_results: int = 100,
) -> pd.DataFrame
```

Поиск прогонов с фильтром MLflow.

```python
from housing.tracking.utils import search_runs

# Найти все прогоны с val_rmse < 4.0
df = search_runs(
    "boston-housing",
    filter_string="metrics.val_rmse < 4.0",
)
```

---

## ClearML: ClearMLTracker (context manager)

Модуль: `src/housing/tracking/clearml.py`

```python
class ClearMLTracker:
    def __init__(
        self,
        cfg: DictConfig,
        stage: str,
        task_type: str = "training",
        extra_tags: list[str] | None = None,
    ) -> None
```

Контекстный менеджер для ClearML задачи. Автоматически инициализирует и закрывает Task.

**Методы:**

| Метод | Описание |
|---|---|
| `connect_dict(name, payload)` | Залогировать словарь параметров |
| `log_metrics(metrics, series, iteration)` | Залогировать scalar-метрики |
| `log_text(message)` | Текстовый лог |
| `upload_artifact(name, artifact)` | Загрузить файл/объект как артефакт |
| `register_model(model_path, model_name, metadata)` | Зарегистрировать модель в ClearML Registry |

**Пример:**

```python
from housing.tracking import ClearMLTracker

with ClearMLTracker(cfg, stage="train", task_type="training") as tracker:
    tracker.connect_dict("train_params", {"alpha": 1.0, "model": "ridge"})
    # ... обучение ...
    tracker.log_metrics(
        metrics={"val_rmse": 4.2, "val_r2": 0.71},
        series="validation",
        iteration=0,
    )
    tracker.upload_artifact("model", "models/ridge.joblib")
    tracker.register_model(
        model_path="models/ridge.joblib",
        model_name="housing-ridge",
        metadata={"val_rmse": 4.2},
    )
```

### `compare_clearml_experiments`

```python
def compare_clearml_experiments(
    project_name: str,
    metric_name: str = "val_rmse",
) -> pd.DataFrame
```

Сравнение всех задач в ClearML-проекте.

### `compare_registered_models`

```python
def compare_registered_models(
    model_name: str,
    project_name: str,
    max_results: int = 50,
) -> pd.DataFrame
```

Сравнение версий модели в ClearML Registry.
