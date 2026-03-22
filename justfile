import 'tasks/check.just'
import 'tasks/install.just'
import 'tasks/clean.just'
import 'tasks/train.just'
import 'tasks/serve.just'

# Show available recipes
default:
    @just --list

# Serve documentation locally
docs:
    uv run mkdocs serve

# Build documentation site
docs-build:
    uv run mkdocs build
