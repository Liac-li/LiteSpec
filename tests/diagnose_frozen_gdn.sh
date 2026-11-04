#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
: "${RUN_GROUP:?Set RUN_GROUP to the completed frozen_gdn run}"
: "${DIAG_INPUTS:?Set DIAG_INPUTS to a directory of independent evaluation JSONL manifests}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export CUDA_VISIBLE_DEVICES="${DIAG_GPU:-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
args=()
for arm in ${EXPERIMENTS:-frozen_reader_legacy_loss frozen_reader_ready_loss frozen_reader_content_only}; do
    args+=(--checkpoint "$arm" "$RUN_GROUP/$arm/last")
done
exec "$PYTHON_BIN" tests/diagnose_gdn.py --run-group "$RUN_GROUP" \
    --inputs-dir "$DIAG_INPUTS" --output-dir "${DIAG_OUTPUT:-$RUN_GROUP/content_diagnostics_$(date +%Y%m%d_%H%M%S)}" \
    --samples-per-dataset "${DIAG_SAMPLES:-16}" --reference-tokens "${DIAG_REFERENCE_TOKENS:-128}" \
    --modes normal off wrong wrong_content content_only scale_only shared_query "${args[@]}" "$@"
