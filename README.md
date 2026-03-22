# Propensity Telecom Case Study

![CI](https://github.com/your-org/propensity-telecom-case-study/actions/workflows/check.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)
![ruff](https://img.shields.io/badge/linter-ruff-orange)

> A company can grow by targeting existing customers. This increases total wallet share and is more cost-effective than acquisition. Through a propensity model we identify the drivers that make a customer most likely to adopt a higher-value plan.

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-org/propensity-telecom-case-study.git
cd propensity-telecom-case-study
uv sync --extra dev

# 2. Generate synthetic data
uv run python scripts/generate_synthetic_data.py

# 3. Train the model (logs to MLflow)
uv run propensity-train

# 4. View experiment results
uv run mlflow ui --backend-store-uri mlruns

# 5. Score a batch of customers
uv run propensity-predict --input data/raw/telecom_customers.csv --output outputs/scores.csv

# 6. Start the REST API server
uv run propensity-serve
# POST /predict  →  {"customers": [...]}
# GET  /health   →  liveness check
```

Or with `just`:
```bash
just install   # set up environment + pre-commit hooks
just train     # train and log to MLflow
just check     # lint + test
just serve     # start the inference API on port 8000
just serve-dev # start with auto-reload (development)
```

## Project Structure

```
propensity-telecom-case-study/
├── src/propensity_telecom_case_study/
│   ├── config.py           # Pydantic schemas + OmegaConf loader
│   ├── scripts.py          # CLI: propensity-train / propensity-predict / propensity-serve
│   ├── domain/             # Pure functions (features, metrics, model)
│   ├── io/                 # Data loaders, MLflow registry
│   ├── application/        # Training and inference pipelines
│   └── api/                # FastAPI inference server (main.py, schemas.py)
├── notebooks/              # EDA and prototype (01_prototype.ipynb)
├── configs/train.yaml      # All hyperparameters and paths
├── tests/                  # Mirrors src/ — 17 tests, 70% coverage
├── scripts/                # One-off utilities (data generation)
├── tasks/                  # just task modules
├── Dockerfile              # Multi-stage production image
└── .github/workflows/      # CI (check.yml) + CD (publish.yml)
```

## Configuration

All model and data settings live in `configs/train.yaml`. Override any value from the CLI:

```bash
uv run propensity-train model.max_depth=10 model.n_estimators=300
```

## Development

```bash
uv sync --extra dev          # install all deps
just install-hooks           # install pre-commit hooks
just check                   # lint + test
just clean                   # remove build artifacts
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.
