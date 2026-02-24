# housing.config - Конфигурация и валидация

Модуль: `src/housing/config/validation.py`

Модуль проверяет корректность конфигурации Hydra перед запуском каждой стадии.

---

## Функции валидации

### `validate_prepare_cfg`

```python
def validate_prepare_cfg(cfg: DictConfig) -> None
```

Проверяет параметры стадии `prepare`.

**Правила:**

- `prepare.test_size` ∈ (0, 1)
- `prepare.random_state` - целое число ≥ 0

**Исключения:** `ValueError` если параметр не валиден.

**Пример:**

```python
from omegaconf import OmegaConf
from housing.config import validate_prepare_cfg

cfg = OmegaConf.create({"prepare": {"test_size": 0.2, "random_state": 42}})
validate_prepare_cfg(cfg)  # OK
```

---

### `validate_train_cfg`

```python
def validate_train_cfg(cfg: DictConfig) -> None
```

Проверяет параметры стадии `train`.

**Правила:**

- `train.val_size` ∈ (0, 1)
- `train.random_state` - целое число ≥ 0
- `model.kind` ∈ {`linear_regression`, `ridge`, `random_forest`}

---

### `validate_selection_cfg`

```python
def validate_selection_cfg(cfg: DictConfig) -> None
```

Проверяет параметры стадии `select_best`.

**Правила:**

- `selection.candidates` - непустой список строк

---

### `validate_evaluate_cfg`

```python
def validate_evaluate_cfg(cfg: DictConfig) -> None
```

Проверяет параметры стадии `evaluate`.

**Правила:**

- `evaluate.model_path` - непустая строка
- `evaluate.test_path` - непустая строка

---

### `validate_clearml_cfg`

```python
def validate_clearml_cfg(cfg: DictConfig) -> None
```

Проверяет параметры ClearML (только если `clearml.enabled=true`).

**Правила:**

- `clearml.project_name` - непустая строка
- `clearml.task_prefix` - непустая строка
- `clearml.server.api_host` - начинается с `http://` или `https://`
- `clearml.server.web_host` - начинается с `http://` или `https://`
- `clearml.server.files_host` - начинается с `http://` или `https://`

---

## Конфигурационные файлы

### `conf/pipeline.yaml` (корневой)

```yaml
defaults:
  - _self_
  - model: linear_regression

prepare:
  test_size: 0.2
  random_state: 42
  raw_path: data/raw/housing.csv
  train_path: data/processed/train.csv
  test_path: data/processed/test.csv

train:
  random_state: 42
  val_size: 0.2
  train_path: data/processed/train.csv

mlflow:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: boston-housing
  model_name: housing-model

clearml:
  enabled: false
  project_name: boston-housing-hw5
  task_prefix: housing-pipeline
  server:
    api_host: http://localhost:8008
    web_host: http://localhost:8080
    files_host: http://localhost:8081
```

### `conf/model/linear_regression.yaml`

```yaml
name: linear_regression
kind: linear_regression
params: {}
```

### `conf/model/ridge.yaml`

```yaml
name: ridge
kind: ridge
params:
  alpha: 1.0
```

### `conf/model/random_forest.yaml`

```yaml
name: random_forest
kind: random_forest
params:
  n_estimators: 100
  max_depth: 10
```
