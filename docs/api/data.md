# housing.data - Загрузка данных

Модуль: `src/housing/data/load.py`

---

## Константы

### `BOSTON_COLUMNS`

```python
BOSTON_COLUMNS: list[str]
```

Список из 14 названий колонок датасета Boston Housing:
`crim`, `zn`, `indus`, `chas`, `nox`, `rm`, `age`, `dis`, `rad`, `tax`, `ptratio`, `b`, `lstat`, `medv`.

---

## Функции

### `load_raw_housing`

```python
def load_raw_housing(path: str | Path | None = None) -> pd.DataFrame
```

Загружает сырой датасет Boston Housing из файла (пробел-разделённый CSV без заголовка).

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `path` | `str \| Path \| None` | `None` | Путь к CSV-файлу. Если `None` - используется `data/raw/housing.csv` относительно корня проекта |

**Возвращает:** `pd.DataFrame` с 14 колонками и ~506 строками.

**Пример:**

```python
from housing.data.load import load_raw_housing

# По умолчанию
df = load_raw_housing()
print(df.shape)  # (506, 14)
print(df.columns.tolist())
# ['crim', 'zn', ..., 'medv']

# Явный путь
df = load_raw_housing("data/raw/housing.csv")
```

---

## Формат данных

| Колонка | Тип | Описание |
|---|---|---|
| `crim` | float | Уровень преступности (на душу населения) |
| `zn` | float | Доля жилой зоны под участки >25 000 кв.фут |
| `indus` | float | Доля нежилых коммерческих акров |
| `chas` | int | 1 если участок граничит с рекой Чарльз |
| `nox` | float | Концентрация оксидов азота (pphm) |
| `rm` | float | Среднее количество комнат в доме |
| `age` | float | Доля домов, построенных до 1940 г. |
| `dis` | float | Средневзвешенное расстояние до 5 центров занятости |
| `rad` | int | Индекс доступности радиальных шоссе |
| `tax` | float | Ставка налога на имущество (на $10 000) |
| `ptratio` | float | Соотношение учеников/учителей по городу |
| `b` | float | 1000(Bk - 0.63)² (Bk - доля афроамериканцев) |
| `lstat` | float | % населения с низким статусом |
| `medv` | float | **Таргет**: медианная стоимость жилья (тыс. $) |
