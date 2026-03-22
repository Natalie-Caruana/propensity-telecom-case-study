# CLAUDE.md

## Commands

```bash
just install          # uv sync --extra dev + pre-commit hooks
just check            # lint + test (run before every commit)
just lint             # ruff check/format + bandit
just test             # pytest --cov=src --cov-fail-under=65
just train            # run training with defaults
just train-custom 6 200  # custom max_depth n_estimators
```

## Architecture (4 layers, top → bottom)

- **config.py** → Pydantic TrainConfig loaded via OmegaConf
- **domain/** → Pure business logic, no I/O (features, model, metrics, drift, explainability, reproducibility)
- **io/** → All side effects (CSV/Parquet, MLflow registry + @champion alias, desktop alerts)
- **application/** → Orchestration only — wires domain + io (training.py, inference.py)

CLI entrypoints (`propensity-train`, `propensity-predict`) registered in `scripts.py` via pyproject.toml.

## Config

All hyperparameters and paths live in `configs/train.yaml`. Override any value at runtime with OmegaConf dot notation:

```bash
uv run propensity-train model.n_estimators=100 data.test_size=0.3
```

Validated by: DataConfig, FeaturesConfig, ModelConfig, MLflowConfig → composed into TrainConfig.

## Rules

- **NEVER** add I/O or side effects to `domain/` — it must stay pure
- `domain/` must **not** import from `io/` or `application/`
- Always use `uv run`, never bare `python` or `pip`
- Run `just check` before suggesting any commit
- Do not modify `configs/train.yaml` defaults without asking

## Tests

- Mirror `src/` structure under `tests/`
- Shared fixtures in `tests/conftest.py` (synthetic DataFrame, temp CSV, minimal TrainConfig)
- Coverage threshold: 65% — enforced in CI and `just test`
- Integration tests in `tests/application/` use a local MLflow tracking server in a temp dir

## Commits

Conventional Commits enforced by commitizen pre-commit hook: `feat:`, `fix:`, `chore:`, `docs:`, `test:`.
Release: `uv run cz bump` → push tags → GitHub Actions builds Docker image + wheel.

<!-- On /compact, preserve: layer boundaries (domain=pure, io=side effects), test conventions, commit format, and the Rules section above. -->
