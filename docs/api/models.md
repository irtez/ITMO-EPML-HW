# housing.models - Обучение моделей

Модуль: `src/housing/models/train.py`

---

## Функции

### `compute_metrics`

```python
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]
```

Вычисляет стандартный набор регрессионных метрик.

**Параметры:**

| Параметр | Тип | Описание |
|---|---|---|
| `y_true` | `np.ndarray` | Истинные значения |
| `y_pred` | `np.ndarray` | Предсказанные значения |

**Возвращает:** `dict` с ключами:

| Ключ | Формула | Описание |
|---|---|---|
| `rmse` | √MSE | Root Mean Squared Error |
| `mae` | mean(\|y − ŷ\|) | Mean Absolute Error |
| `r2` | 1 − SS_res/SS_tot | Коэффициент детерминации |

**Пример:**

```python
from housing.models.train import compute_metrics
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

metrics = compute_metrics(y_true, y_pred)
print(metrics)
# {'rmse': 0.612, 'mae': 0.5, 'r2': 0.948}
```

---

### `train_and_log_model`

```python
def train_and_log_model(
    model_name: str,
    model: BaseEstimator,
    feature_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    extra_params: dict[str, Any] | None = None,
) -> tuple[str, float]
```

Обучает модель, собирает метрики и логирует всё в MLflow.

**Параметры:**

| Параметр | Тип | Описание |
|---|---|---|
| `model_name` | `str` | Название для MLflow run |
| `model` | `BaseEstimator` | Sklearn-совместимая модель |
| `feature_pipeline` | `Pipeline` | Sklearn Pipeline для трансформации признаков |
| `X_train` | `pd.DataFrame` | Обучающие признаки |
| `y_train` | `pd.Series` | Обучающий таргет |
| `X_val` | `pd.DataFrame` | Валидационные признаки |
| `y_val` | `pd.Series` | Валидационный таргет |
| `extra_params` | `dict \| None` | Дополнительные параметры для логирования в MLflow |

**Действия:**

1. Строит полный Pipeline: `feature_pipeline → model`
2. Обучает на `(X_train, y_train)`
3. Вычисляет метрики на train и val
4. Логирует в MLflow: параметры, метрики, артефакт модели

**Возвращает:** Кортеж `(run_id, val_rmse)`

**Пример:**

```python
from sklearn.ensemble import RandomForestRegressor
from housing.features.build import get_feature_pipeline
from housing.models.train import train_and_log_model

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
feature_pipeline = get_feature_pipeline()

run_id, val_rmse = train_and_log_model(
    model_name="random_forest",
    model=model,
    feature_pipeline=feature_pipeline,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    extra_params={"n_estimators": 100, "max_depth": 10},
)
print(f"Run: {run_id}, Val RMSE: {val_rmse:.4f}")
```

---

## Поддерживаемые модели

| `kind` | Класс sklearn | Гиперпараметры |
|---|---|---|
| `linear_regression` | `LinearRegression` | - |
| `ridge` | `Ridge` | `alpha` |
| `random_forest` | `RandomForestRegressor` | `n_estimators`, `max_depth` |

Создание модели по `kind` выполняется в `scripts/train.py`.
