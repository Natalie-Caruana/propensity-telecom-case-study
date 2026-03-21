# Clean generated artifacts

clean:
    uv run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['dist', 'build', '.cache', 'htmlcov', '.pytest_cache', '.ruff_cache', '.mypy_cache']]"
    uv run python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
    uv run python -c "import pathlib; [p.unlink(missing_ok=True) for p in [pathlib.Path('.coverage')]]"

clean-mlflow:
    uv run python -c "import shutil; shutil.rmtree('mlruns', ignore_errors=True); shutil.rmtree('mlartifacts', ignore_errors=True)"
