# Getting Started

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — package manager
- [just](https://just.systems/) — task runner (optional)

## Installation

```bash
git clone https://github.com/your-org/propensity-telecom-case-study.git
cd propensity-telecom-case-study
uv sync --extra dev
```

## Generate Data

```bash
uv run python scripts/generate_synthetic_data.py
# → data/raw/telecom_customers.csv (600 rows)
```

## Train

```bash
uv run propensity-train
# or override hyperparameters:
uv run propensity-train model.max_depth=10 model.n_estimators=300
```

## View Experiments

```bash
uv run mlflow ui --backend-store-uri mlruns
# open http://localhost:5000
```

## Score Customers

```bash
uv run propensity-predict \
  --input data/raw/telecom_customers.csv \
  --output outputs/scores.csv
```

## Run Checks

```bash
just check   # lint + test
just test    # tests only
just lint    # ruff + bandit only
```
