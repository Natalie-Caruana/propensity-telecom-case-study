"""CLI entrypoints — registered in pyproject.toml [project.scripts]."""

import sys

from loguru import logger

from propensity_telecom_case_study.application.inference import run_inference
from propensity_telecom_case_study.application.training import run_training
from propensity_telecom_case_study.config import load_config


def train() -> None:
    """Train the propensity model and log results to MLflow.

    Usage:
        uv run propensity-train
        uv run propensity-train model.max_depth=8 model.n_estimators=300
    """
    cfg = load_config()
    logger.info(f"Starting training — experiment: {cfg.mlflow.experiment_name}")
    metrics = run_training(cfg)
    roc = metrics["roc_auc"]
    pr = metrics["avg_precision"]
    logger.info(f"Training complete. ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}")


def predict() -> None:
    """Score a batch of customers using the latest registered model.

    Usage:
        uv run propensity-predict --input data/raw/telecom_customers.csv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Score customers with the propensity model."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--output", default="outputs/scores.csv", help="Output path.")
    args = parser.parse_args()

    cfg = load_config()
    scored = run_inference(cfg, args.input)

    import pathlib

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    logger.info(f"Scores written to {out}")


def serve() -> None:
    """Start the FastAPI inference server.

    Usage:
        uv run propensity-serve
        uv run propensity-serve --host 0.0.0.0 --port 8080
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Propensity inference server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")  # nosec B104
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    args = parser.parse_args()

    uvicorn.run(
        "propensity_telecom_case_study.api.main:app",
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    sys.exit(train())
