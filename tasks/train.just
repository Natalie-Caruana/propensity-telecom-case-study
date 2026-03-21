# ML training tasks

train:
    uv run propensity-train

train-custom max_depth n_estimators:
    uv run propensity-train model.max_depth={{max_depth}} model.n_estimators={{n_estimators}}

mlflow-ui:
    uv run mlflow ui --backend-store-uri mlruns
