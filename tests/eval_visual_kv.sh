#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${RUN_GROUP:?Set RUN_GROUP to the completed training group}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv-vlm/bin/python}"
exec "$PYTHON_BIN" tests/evaluate_visual_kv.py --run-group "$RUN_GROUP" "$@"
