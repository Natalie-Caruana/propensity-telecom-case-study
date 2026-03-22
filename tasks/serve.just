# Start the FastAPI inference server (default: port 8000)
serve:
    uv run propensity-serve

# Start with auto-reload for development
serve-dev:
    uv run uvicorn propensity_telecom_case_study.api.main:app --reload --port 8000
