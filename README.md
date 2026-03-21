# Propensity Telecom Case Study

A propensity model to identify the drivers that make existing telecom customers most likely to adopt a higher-value plan — increasing total wallet share in a cost-effective way.

## Quickstart

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Run training pipeline
propensity-train
```

## Project Structure

```
src/propensity_telecom_case_study/
├── domain/       # Pure functions: feature engineering, metrics
├── io/           # Data loaders and model serialization
├── application/  # Pipeline orchestration
└── scripts.py    # CLI entrypoints

notebooks/        # Exploratory analysis and prototyping
configs/          # OmegaConf YAML configuration files
tests/            # Mirrors src/ structure
```
