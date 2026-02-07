# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==2.3.2

# Copy dependency files first (better layer caching)
COPY pyproject.toml poetry.lock ./

# Install only production dependencies (no dev tools) into a virtualenv
RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-interaction --no-ansi --no-root

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy the pre-built virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy project source code and pipeline files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY params.yaml dvc.yaml ./

# Copy raw data directly (DVC-managed file included explicitly for the image)
COPY data/raw/housing.csv ./data/raw/housing.csv

# Create directories needed at runtime
RUN mkdir -p models metrics mlruns data/processed

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="file:./mlruns" \
    GIT_PYTHON_REFRESH=quiet

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command: run the full pipeline (prepare → train → evaluate)
CMD ["sh", "-c", "python scripts/prepare.py && python scripts/train.py && python scripts/evaluate.py"]
