#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r docs/requirements.txt >/dev/null
sphinx-build -b html docs docs/_build/html
