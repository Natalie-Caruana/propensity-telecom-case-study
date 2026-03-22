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
# GET  /         →  scoring UI (open in browser)
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

## Inference API & UI

Start the server:

```bash
uv run propensity-serve              # default: http://localhost:8000
uv run propensity-serve --port 8080  # custom port
just serve-dev                       # auto-reload for development
```

| Route | Method | Description |
|---|---|---|
| `/` | GET | Browser scoring UI — fill in customer fields and get propensity scores |
| `/predict` | POST | JSON batch inference — accepts `{"customers": [...]}` |
| `/health` | GET | Liveness check — confirms model is loaded |
| `/docs` | GET | Auto-generated OpenAPI / Swagger UI |

### Scoring UI

Open `http://localhost:8000` in a browser after starting the server.

- Add one or more customer records (tabbed)
- Fill in the 13 feature fields (numeric, categorical, binary)
- Click **Score** to call `/predict` and see colour-coded propensity results:
  - **Green** ≥ 65% — high propensity
  - **Orange** 35–65% — medium propensity
  - **Red** < 35% — low propensity

The API URL field at the top of the page can be pointed at any running instance of the server.

### REST example

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [{
      "age": 35, "tenure_months": 24, "monthly_charges": 65,
      "data_usage_gb": 10, "call_minutes": 200,
      "num_products": 2, "num_complaints": 0, "customer_service_calls": 1,
      "region": "North", "contract_type": "monthly",
      "internet_service": "fiber", "has_streaming": 1, "has_device_protection": 0
    }]
  }' | python -m json.tool
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
│   └── api/                # FastAPI inference server (main.py, schemas.py, static/)
├── notebooks/              # EDA and prototype (01_prototype.ipynb)
├── configs/train.yaml      # All hyperparameters and paths
├── tests/                  # Mirrors src/ — 22 tests, 80% coverage
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
