# DS project template

Boston housing price analysis and prediction (ITMO EPML Course Homework).

## Project structure

```
ITMO-EPML-HW/
├── data/
│   ├── raw/            # Original, immutable data
│   ├── interim/        # Intermediate transformed data
│   ├── processed/      # Final data ready for modeling
│   └── external/       # Third-party external data
├── models/             # Serialized trained models
├── notebooks/          # Jupyter notebooks for EDA & experiments
├── reports/
│   └── figures/        # Generated charts and visualizations
├── conf/               # Hydra configs (composition + overrides)
│   └── model/          # Per-algorithm configs
├── src/
│   └── housing/        # Installable Python package
│       ├── config/     # Config validation helpers
│       ├── data/       # Data loading & cleaning scripts
│       ├── features/   # Feature engineering
│       ├── models/     # Model training & evaluation
│       ├── pipeline/   # Monitoring and notifications
│       ├── tracking/   # MLflow tracking utilities (decorator, context manager, utils)
│       └── visualization/
├── scripts/
│   ├── prepare.py      # DVC stage 1: raw → train/test split
│   ├── train.py        # DVC stages 2-4: per-model training
│   ├── select_best.py  # DVC stage 5: best model selection + registry
│   ├── evaluate.py     # DVC stage 6: evaluation on test set
│   └── run_experiments.py  # Standalone: run 21 HW3 experiments
├── template/           # Cookiecutter template for new DS projects
├── tests/              # pytest test suite
├── .pre-commit-config.yaml
├── Dockerfile
└── pyproject.toml
```

## Quickstart

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd ITMO-EPML-HW

# Install all dependencies (creates .venv automatically)
poetry install

# Pull data from DVC remote storage
poetry run dvc pull

# Activate the virtual environment
poetry shell
```

### Using the Cookiecutter template

To scaffold a new DS project from the included template:

```bash
cookiecutter template/
```

## Git workflow

| Branch | Purpose |
|--------|---------|
| `master` | Stable: contains only completed, submitted homework |
| `hw/hw1` | HW1 branch (points to the same commit as `master`) |
| `hw/hwN` | Working branch for each new homework, branched from `master` |

```bash
# Start a new homework
git checkout -b hw/hw2 master

# After completion — merge back with an explicit merge commit
git checkout master
git merge --no-ff hw/hw2 -m "feat: homework 2 — <topic>"
```

## Development

### Code quality

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
(replaces Black + isort + Flake8), [MyPy](https://mypy.readthedocs.io/) for type checking, and [Bandit](https://bandit.readthedocs.io/) for security analysis.

```bash
# Format code
poetry run ruff format .

# Lint (auto-fix where possible)
poetry run ruff check . --fix

# Type check
poetry run mypy src/

# Security scan
poetry run bandit -c pyproject.toml -r src/
```

### Pre-commit hooks

Hooks run automatically on every `git commit`. Install them once:

```bash
poetry run pre-commit install
```

Run all hooks manually:

```bash
poetry run pre-commit run --all-files
```

### Tests

```bash
poetry run pytest
```

## ML Pipeline (DVC)

Data and pipeline are managed with [DVC](https://dvc.org/). Raw data is stored in a local DVC remote (`~/dvc-storage`).

```bash
# Download data
poetry run dvc pull

# Run the full HW4 pipeline
poetry run dvc repro

# Run multiple experiments in parallel (DVC queue workers)
poetry run dvc exp run --queue train_linear_regression
poetry run dvc exp run --queue train_ridge
poetry run dvc exp run --queue train_random_forest
poetry run dvc queue start -j 3

# Show metrics
poetry run dvc metrics show
```

HW4 pipeline graph:
`prepare -> {train_linear_regression, train_ridge, train_random_forest} -> select_best -> evaluate`

The pipeline is defined in `dvc.yaml` and uses Hydra composition from `conf/pipeline.yaml` + `conf/model/*.yaml`.

## Configuration Management (Hydra)

Hydra is used for config composition, algorithm switching, and CLI overrides.

```bash
# Default config
poetry run python scripts/train.py

# Override algorithm config
poetry run python scripts/train.py model=random_forest

# Override any value from CLI
poetry run python scripts/train.py model=ridge model.params.alpha=0.5
```

Validation rules are implemented in `src/housing/config/validation.py`.

## Experiment Tracking (MLflow)

Experiments are tracked locally via MLflow with a SQLite backend (`mlflow.db`, not committed to git):

```bash
# Run the extended HW3 experiment suite (21 runs)
poetry run python scripts/run_experiments.py

# Launch the MLflow UI
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Open http://localhost:5000
```

The `src/housing/tracking/` module provides three integration patterns:
- `@mlflow_run` — decorator for wrapping a function in a run
- `ExperimentTracker` — context manager with `log_param/s`, `log_metric/s`, `set_tag`
- `compare_runs`, `get_best_run`, `search_runs` — utilities for querying results

## Docker

```bash
# Build the image
docker build -t housing:hw3 .

# Run the full pipeline inside the container
docker run --rm housing:hw3
```
