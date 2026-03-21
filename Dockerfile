# ── Build stage ───────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Layer cache: install deps before copying source
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the package itself
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Copy only the virtual environment from the build stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Add .venv to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default entrypoint: training pipeline
ENTRYPOINT ["propensity-train"]
