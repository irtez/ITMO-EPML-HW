# Воспроизводимость

В проекте применяется многоуровневый подход к воспроизводимости.

---

## 1. Воспроизведение с нуля

Полная последовательность команд для воспроизведения всех результатов:

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd ITMO-EPML-HW

# 2. Установить Python-зависимости (фиксированные версии через poetry.lock)
poetry install

# 3. Получить данные из DVC remote
poetry run dvc pull

# 4. Запустить весь пайплайн
poetry run dvc repro

# 5. Посмотреть метрики
poetry run dvc metrics show

# 6. (Опционально) Запустить расширенные эксперименты
poetry run python scripts/run_experiments.py

# 7. Проверить тесты
poetry run pytest

# 8. Сгенерировать отчёт
poetry run python scripts/generate_report.py
```

Ожидаемые результаты (test_metrics.json):

```json
{
  "rmse": 3.449,
  "mae": 2.269,
  "r2": 0.838
}
```

---

## 2. Фиксация зависимостей

### Python-пакеты (Poetry)

`poetry.lock` фиксирует **точные версии** всех транзитивных зависимостей:

```bash
# Установить строго из lock-файла (без обновления)
poetry install --no-root

# Проверить, что lock-файл актуален
poetry check
```

### Данные (DVC)

`data/raw/housing.csv.dvc` - это DVC-указатель с хешем файла:

```yaml
outs:
- md5: abc123...
  size: 50260
  path: housing.csv
```

Команда `dvc pull` скачивает **именно тот** файл с тем же хешем.

```bash
# Проверить статус данных
poetry run dvc status

# Просмотреть граф зависимостей
poetry run dvc dag
```

---

## 3. Воспроизводимость результатов

### Random state

Все случайные операции используют фиксированный seed `42`:

```python
# В conf/pipeline.yaml
prepare:
  random_state: 42

train:
  random_state: 42
```

### Версионирование моделей

`dvc.lock` фиксирует хеши входных и выходных файлов каждой стадии:

```bash
# Проверить, что стадии актуальны
poetry run dvc status

# Если всё актуально - нет изменений для воспроизведения
# Running stages 'prepare': changed.
```

---

## 4. Docker (изолированное окружение)

Для полной изоляции используйте Docker:

```bash
# Сборка образа
docker build -t housing:latest .

# Запуск пайплайна в контейнере
docker run --rm housing:latest

# Проверить метрики из контейнера
docker run --rm housing:latest cat metrics/test_metrics.json
```

Dockerfile использует:
- Фиксированные базовые образы (`python:3.10-slim`)
- Фиксированную версию Poetry (`2.3.2`)
- Копирование `poetry.lock` для детерминированной установки зависимостей

---

## 5. ClearML (MLOps воспроизводимость)

ClearML дополнительно фиксирует:

- Гиперпараметры задачи
- Версии кода (git commit hash)
- Артефакты (train/test датасеты, обученные модели)
- Окружение (Python packages snapshot)

Воспроизведение через ClearML:

```bash
# Запустить стадию с полным трекингом
poetry run python scripts/train.py model=random_forest clearml.enabled=true

# Воспроизвести задачу по ID
# (в ClearML UI: Task → Clone → Enqueue)
```

---

## 6. Конфигурация (Hydra)

Каждый запуск конфигурируется через Hydra - параметры явно передаются в команде:

```bash
# Полностью воспроизводимый запуск (все параметры явно)
poetry run python scripts/train.py \
    model=random_forest \
    model.params.n_estimators=100 \
    model.params.max_depth=10 \
    train.random_state=42 \
    train.val_size=0.2 \
    prepare.test_size=0.2
```

---

## 7. Проверка воспроизводимости

```bash
# Тест воспроизводимости
poetry run pytest tests/test_reproducibility.py -v

# Полный набор тестов
poetry run pytest --cov=src/housing

# Проверить метрики через DVC
poetry run dvc metrics show --all-commits
```

---

## Чеклист воспроизводимости

- [x] `poetry.lock` - точные версии Python-зависимостей
- [x] `dvc.lock` - хеши данных и артефактов пайплайна
- [x] `*.dvc` файлы - DVC-указатели на файлы данных
- [x] `params.yaml` + `conf/` - все параметры зафиксированы
- [x] `random_state=42` - фиксированный seed везде
- [x] `Dockerfile` - фиксированные образы и версии
- [x] `pytest` - тесты подтверждают корректность кода
- [x] ClearML task - snapshot окружения и артефактов
