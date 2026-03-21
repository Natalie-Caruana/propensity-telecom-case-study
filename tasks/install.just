# Environment setup

install:
    uv sync --extra dev

install-hooks:
    uv run pre-commit install
    uv run pre-commit install --hook-type commit-msg
