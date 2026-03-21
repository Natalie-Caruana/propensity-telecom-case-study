# Contributing

Thank you for your interest in contributing! This guide covers environment setup, the PR process, and testing standards.

## Environment Setup

```bash
git clone https://github.com/your-org/propensity-telecom-case-study.git
cd propensity-telecom-case-study
uv sync --extra dev
just install-hooks
```

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), [just](https://just.systems/)

## Branching

We use **GitHub Flow**:

1. Fork or create a feature branch from `main`: `git checkout -b feat/my-feature`
2. Commit using [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `chore:` tooling/config
   - `docs:` documentation only
   - `test:` tests only
3. Open a PR against `main` — CI must pass before merge

## Running Checks

```bash
just check        # lint + format check + security scan + tests
just lint         # ruff check + bandit only
just test         # pytest with coverage
```

All checks run automatically as pre-commit hooks on every commit.

## Testing Standards

- Mirror `src/` in `tests/` — one test file per module
- Use **Given / When / Then** comments in each test
- Shared fixtures go in `tests/conftest.py`
- Coverage threshold: **65%** (enforced in CI)

## Adding a New Model

1. Add your model config fields to `ModelConfig` in `src/.../config.py`
2. Update `build_pipeline()` in `src/.../domain/model.py`
3. Update `configs/train.yaml` with sensible defaults
4. Add tests in `tests/domain/test_model.py`

## Commit Message Format

Enforced by `commitizen` pre-commit hook:

```
<type>(<optional scope>): <short summary>

[optional body]
```

Example: `feat(model): add gradient boosting classifier option`

## Release Process

1. Bump version: `uv run cz bump`
2. Review and push: `git push --follow-tags`
3. Create a GitHub Release — the `publish.yml` workflow builds and pushes the Docker image automatically
