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
├── src/
│   └── housing/        # Installable Python package
│       ├── data/       # Data loading & cleaning scripts
│       ├── features/   # Feature engineering
│       ├── models/     # Model training & evaluation
│       └── visualization/
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

## Docker

```bash
# Build the image
docker build -t housing .

# Run a container
docker run --rm housing
```
