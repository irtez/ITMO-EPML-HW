# housing.features - Инженерия признаков

Модуль: `src/housing/features/build.py`

---

## Константы

### `FEATURE_COLUMNS`

```python
FEATURE_COLUMNS: list[str]
```

Список 13 признаков: `['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']`

### `TARGET_COLUMN`

```python
TARGET_COLUMN: str  # = "medv"
```

---

## Функции

### `get_feature_pipeline`

```python
def get_feature_pipeline() -> Pipeline
```

Создаёт sklearn Pipeline для трансформации признаков.

**Текущий пайплайн:** одна ступень - `StandardScaler` (нормализация).

**Возвращает:** `sklearn.pipeline.Pipeline`

**Пример:**

```python
from housing.features.build import get_feature_pipeline

feature_pipeline = get_feature_pipeline()
print(feature_pipeline.steps)
# [('scaler', StandardScaler())]

X_scaled = feature_pipeline.fit_transform(X_train)
```

---

### `split_features_target`

```python
def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]
```

Разделяет DataFrame на матрицу признаков X и вектор таргета y.

**Параметры:**

| Параметр | Тип | Описание |
|---|---|---|
| `df` | `pd.DataFrame` | DataFrame с колонками признаков и таргета |

**Возвращает:** Кортеж `(X, y)`:
- `X: pd.DataFrame` - только колонки из `FEATURE_COLUMNS` (13 колонок)
- `y: pd.Series` - колонка `TARGET_COLUMN` (`medv`)

**Пример:**

```python
from housing.data.load import load_raw_housing
from housing.features.build import split_features_target

df = load_raw_housing()
X, y = split_features_target(df)
print(X.shape)   # (506, 13)
print(y.shape)   # (506,)
print(y.name)    # 'medv'
```

---

## Пример полного пайплайна

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from housing.data.load import load_raw_housing
from housing.features.build import get_feature_pipeline, split_features_target

# Загрузить данные
df = load_raw_housing()
X, y = split_features_target(df)

# Разделить на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Получить пайплайн и трансформировать
feature_pipeline = get_feature_pipeline()
X_train_scaled = feature_pipeline.fit_transform(X_train)
X_test_scaled = feature_pipeline.transform(X_test)
```
