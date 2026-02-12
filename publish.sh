#!/bin/bash
set -e

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build sdist and wheel
python -m build

# Upload to PyPI
twine upload dist/*
