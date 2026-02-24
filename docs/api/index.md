# API Reference

Пакет `housing` (`src/housing/`) содержит все переиспользуемые модули проекта.

## Структура пакета

```
src/housing/
├── config/          # Валидация конфигурации Hydra
├── data/            # Загрузка данных
├── features/        # Инженерия признаков
├── models/          # Обучение и оценка моделей
├── pipeline/        # Мониторинг стадий
└── tracking/        # MLflow и ClearML трекинг
```

## Импорты

```python
# Данные
from housing.data.load import load_raw_housing

# Признаки
from housing.features.build import (
    get_feature_pipeline,
    split_features_target,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

# Модели
from housing.models.train import compute_metrics, train_and_log_model

# Трекинг
from housing.tracking import (
    ExperimentTracker,   # MLflow context manager
    mlflow_run,          # MLflow decorator
    ClearMLTracker,      # ClearML context manager
)
from housing.tracking.utils import compare_runs, get_best_run, search_runs

# Конфиг
from housing.config import (
    validate_prepare_cfg,
    validate_train_cfg,
    validate_selection_cfg,
    validate_evaluate_cfg,
    validate_clearml_cfg,
)
```

## Разделы

| Модуль | Ссылка |
|---|---|
| `housing.data` | [Загрузка данных](data.md) |
| `housing.features` | [Инженерия признаков](features.md) |
| `housing.models` | [Обучение моделей](models.md) |
| `housing.tracking` | [Трекинг экспериментов](tracking.md) |
| `housing.config` | [Конфигурация](config.md) |
