# Configuration

All settings live in `configs/train.yaml`. Override any value from the CLI without touching the file.

## Full Config Reference

```yaml
data:
  raw_path: "data/raw/telecom_customers.csv"
  test_size: 0.2          # fraction held out for evaluation
  random_state: 42

features:
  numeric: [age, tenure_months, monthly_charges, ...]
  categorical: [region, contract_type, internet_service]
  binary: [has_streaming, has_device_protection]
  target: upgraded

model:
  n_estimators: 200       # number of trees
  max_depth: 6            # tree depth
  min_samples_leaf: 5     # min samples per leaf (regularisation)
  class_weight: balanced  # handles class imbalance
  random_state: 42

mlflow:
  experiment_name: propensity-telecom
  model_name: propensity-rf
  tracking_uri: "mlruns"
```

## CLI Overrides

```bash
uv run propensity-train model.max_depth=8 model.n_estimators=500
```

OmegaConf merges CLI args over the YAML, so only the specified keys change.
