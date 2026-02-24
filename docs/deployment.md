# Руководство по развертыванию

## Предварительные требования

| Инструмент | Версия | Ссылка |
|---|---|---|
| Python | 3.10+ | [python.org](https://python.org) |
| Poetry | 1.8+ | [python-poetry.org](https://python-poetry.org) |
| Git | любая | - |
| Docker + Compose | 24+ | [docker.com](https://docker.com) _(только для ClearML)_ |

---

## 1. Клонирование и установка зависимостей

```bash
# Клонировать репозиторий
git clone <repo-url>
cd ITMO-EPML-HW

# Установить зависимости (Poetry создаёт .venv автоматически)
poetry install

# Активировать виртуальное окружение
poetry shell
```

---

## 2. Данные (DVC)

Данные хранятся в DVC-remote (локальная папка `~/dvc-storage`).

```bash
# Скачать данные из DVC remote
poetry run dvc pull

# Проверить структуру данных
ls data/raw/   # housing.csv
ls data/processed/   # train.csv, test.csv (после dvc repro)
```

!!! note "DVC remote"
    При первом развёртывании убедитесь, что DVC remote настроен:
    ```bash
    poetry run dvc remote list
    ```

---

## 3. Запуск ML-пайплайна

```bash
# Запустить весь пайплайн (6 стадий)
poetry run dvc repro

# Проверить метрики
poetry run dvc metrics show
```

Пайплайн: `prepare → train_{lr,ridge,rf} → select_best → evaluate`

---

## 4. MLflow (трекинг экспериментов)

MLflow работает локально с SQLite бэкендом (файл `mlflow.db` не коммитится в git).

```bash
# Запустить UI
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Открыть: http://localhost:5000
```

Запуск расширенного набора экспериментов (HW3, 21 прогон):

```bash
poetry run python scripts/run_experiments.py
```

---

## 5. ClearML (MLOps-платформа)

### 5.1 Поднять ClearML Server

```bash
cd infra/clearml
cp .env.example .env
# Отредактировать .env (API keys)
docker compose up -d
```

После запуска:

| Сервис | URL |
|---|---|
| Web UI | http://localhost:8080 |
| API | http://localhost:8008 |
| File server | http://localhost:8081 |

### 5.2 Настроить клиент

```bash
cd ../..
cp conf/clearml/clearml.conf.example clearml.conf
export CLEARML_API_ACCESS_KEY=<your-key>
export CLEARML_API_SECRET_KEY=<your-secret>
export CLEARML_CONFIG_FILE=$PWD/clearml.conf
```

### 5.3 Запустить с ClearML

```bash
# Включить ClearML трекинг (в conf/pipeline.yaml: clearml.enabled: true)
poetry run dvc repro

# Или запустить ClearML Pipeline Controller
poetry run python scripts/clearml_pipeline.py clearml.enabled=true
```

### 5.4 ClearML Agent (опционально)

```bash
cd infra/clearml
docker compose --profile agent up -d clearml-agent
```

---

## 6. Docker (автономный запуск пайплайна)

```bash
# Собрать образ
docker build -t housing:latest .

# Запустить пайплайн внутри контейнера
docker run --rm housing:latest
```

Образ: multi-stage build (builder + runtime), запуск от non-root пользователя.

---

## 7. Документация (MkDocs)

```bash
# Установить зависимости для документации
pip install mkdocs mkdocs-material

# Запустить локально
mkdocs serve
# Открыть: http://127.0.0.1:8000

# Собрать статику
mkdocs build

# Задеплоить на GitHub Pages
mkdocs gh-deploy --force
```

---

## 8. Инструменты разработки

```bash
# Форматирование
poetry run ruff format .

# Линтинг (с автоисправлением)
poetry run ruff check . --fix

# Проверка типов
poetry run mypy src/

# Безопасность
poetry run bandit -c pyproject.toml -r src/

# Тесты с покрытием
poetry run pytest

# Pre-commit хуки (один раз)
poetry run pre-commit install

# Запустить все хуки вручную
poetry run pre-commit run --all-files
```
