#!/usr/bin/env bash
# Format colbench/ in Google style, matching what gpylint expects in google3.
#
# colbench/ is deliberately scoped out of verl's ruff config (see pyproject.toml
# [tool.ruff] extend-exclude) because this tree is mirrored into google3, where the
# house style is 2-space indent at 80 columns rather than verl's 4-space at 120.
#
# Use pyink, NOT `yapf --style=google`: yapf's google style follows the *public*
# Google Python Style Guide, which specifies 4-space indent. The 2-space convention is
# google3-internal and pyink only applies it when --pyink-indentation 2 is passed.
#
# Note pyink pins black<26, so install it somewhere isolated if you also use black:
#   pip install pyink
set -euo pipefail

cd "$(dirname "$0")/.."

# Every .py under colbench/, including selfplay/ and tests/ -- `colbench/*.py` would
# silently cover only the 14 top-level files out of 34.
mapfile -t files < <(find colbench -name '*.py' -not -path '*__pycache__*')

pyink --line-length 80 --pyink-indentation 2 "${files[@]}" "$@"
