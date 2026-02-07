# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==2.3.2

# Copy dependency files first (better layer caching)
COPY pyproject.toml poetry.lock ./

# Install only production dependencies (no dev tools) into a virtualenv
RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-interaction --no-ansi

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy the pre-built virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy project source code
COPY src/ ./src/
COPY data/ ./data/

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command: open an interactive Python shell with the project available
CMD ["python", "-c", "from housing import *; print('Housing package loaded successfully')"]
