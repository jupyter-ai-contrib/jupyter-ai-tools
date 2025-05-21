#!/usr/bin/env bash
set -eux  # exit on error and print commands

# Type checking
mypy .

# Linting
ruff check .

# Formatting (just checks, doesn't modify)
black --check --diff .

# Markdown formatting (optional)
mdformat --check *.md || true  # only run if you use README.md

# Unit tests
pytest --rootdir=. --disable-warnings

