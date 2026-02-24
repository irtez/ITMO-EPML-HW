# Отчёт об экспериментах - 24.02.2026 23:56:06

> Автоматически сгенерировано скриптом `scripts/generate_report.py`
> Данные: `metrics/` (выходы DVC-пайплайна)

---

## 1. Конфигурация пайплайна

| Параметр | Значение |
|---|---|
| Test size | 0.2 |
| Val size | 0.2 |
| Random state | 42 |
| Метрика выбора | Val RMSE (минимизация) |

> Полная конфигурация: `conf/pipeline.yaml`, `conf/model/*.yaml`

---

## 2. Результаты обучения

### 2.1 Сравнительная таблица моделей

| Модель | Train RMSE | Val RMSE | Val MAE | Val R² | Итог |
|---|---|---|---|---|---|
| Linear Regression | 4.6269 | 4.8256 | 3.2108 | 0.6725 | - |
| Ridge (α=1.0) | 4.6271 | 4.8248 | 3.2020 | 0.6726 | - |
| Random Forest | 1.6505 | 3.7762 | 2.4989 | 0.7994 | ✅ **Лучшая** |


![Сравнение моделей](figures/model_comparison_example.png)

### 2.2 Параметры моделей

| Модель | Гиперпараметры |
|---|---|
| Linear Regression | - |
| Ridge (α=1.0) | alpha = 1.0 |
| Random Forest | n_estimators = 100, max_depth = 10 |

> Конфигурация: `conf/model/*.yaml`

---

## 3. Анализ переобучения (train vs val RMSE)

| Модель | Train RMSE | Val RMSE | Gap (Val − Train) | Оценка |
|---|---|---|---|---|
| Linear Regression | 4.6269 | 4.8256 | **0.1987** | Минимальное |
| Ridge (α=1.0) | 4.6271 | 4.8248 | **0.1978** | Минимальное |
| Random Forest | 1.6505 | 3.7762 | **2.1256** | Умеренное |


- **Linear Regression**: gap = 0.1987 → минимальное переобучение
- **Ridge (α=1.0)**: gap = 0.1978 → минимальное переобучение
- **Random Forest**: gap = 2.1256 → умеренное переобучение

![Train vs Val RMSE](figures/train_val_gap_example.png)

---

## 4. Результаты на тестовой выборке

Лучшая модель по валидации - **Random Forest** - оценена на отложенной тестовой выборке:

| Метрика | Значение |
|---|---|
| **RMSE** | **3.4494** |
| **MAE** | **2.2692** |
| **R²** | **0.8378** |

![Тестовые метрики](figures/test_metrics_example.png)

---

## 5. Итоговые показатели

1. **Лучшая модель:** Random Forest - val RMSE = 3.7762
2. **Снижение val RMSE** относительно худшей (Linear Regression): **21.7%**
3. **Наименьший gap** train/val: Ridge (α=1.0) (gap = 0.1978)
4. **Наибольший gap** train/val: Random Forest (gap = 2.1256)
5. **Тестовые метрики** (Random Forest): RMSE = 3.4494, MAE = 2.2692, R² = 0.8378

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
