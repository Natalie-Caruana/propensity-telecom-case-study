# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-21

### Added
- Project initialization with `uv`, `git`, and VS Code settings
- Synthetic telecom customer dataset generator (600 rows, 14 features)
- EDA and baseline model prototype notebook (`notebooks/01_prototype.ipynb`)
  - Target distribution analysis
  - Dummy vs Logistic Regression vs Random Forest comparison
  - ROC curve, confusion matrix, feature importance plots
- Production `src/` package with domain / io / application layers
  - `config.py` — Pydantic + OmegaConf configuration management
  - `domain/` — pure feature engineering, metrics, model construction
  - `io/` — CSV loader, MLflow model registry with `@champion` alias
  - `application/` — training and inference pipelines
  - `scripts.py` — `propensity-train` and `propensity-predict` CLI entrypoints
- Validation layer: 17 tests, 70% coverage, ruff + bandit clean
- Automation: `justfile`, pre-commit hooks, Dockerfile, GitHub Actions CI/CD
- `configs/train.yaml` — all hyperparameters and paths in one place

[Unreleased]: https://github.com/your-org/propensity-telecom-case-study/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/propensity-telecom-case-study/releases/tag/v0.1.0
