# Quality checks

check: lint test

lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/
    uv run bandit -r src/ -q

test:
    uv run pytest tests/ -v --cov=src --cov-fail-under=65
