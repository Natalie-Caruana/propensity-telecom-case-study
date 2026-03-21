# Propensity Telecom Case Study

A company can grow by targeting existing customers. This increases total wallet share and is more cost-effective than acquisition. Through a propensity model we identify the drivers that make a customer most likely to adopt a higher-value plan.

## Key Features

- **End-to-end MLOps pipeline** — from raw data to registered model
- **Reproducible** — locked dependencies, fixed random seeds, Docker image
- **Observable** — structured logging, MLflow experiment tracking, `@champion` model alias
- **Validated** — 17 tests, ruff + bandit clean, CI on every PR

## Architecture

```
Raw CSV → DatasetLoader → train/test split
                              ↓
                     ColumnTransformer (scale + encode)
                              ↓
                     RandomForestClassifier
                              ↓
                     MLflow (params + metrics + model @champion)
```

## Quick Links

- [Getting Started](getting-started.md)
- [Configuration](configuration.md)
- [Contributing](contributing.md)
- [Changelog](changelog.md)
