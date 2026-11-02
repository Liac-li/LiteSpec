#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${RUN_GROUP:?Set RUN_GROUP to the existing visual-kv training output group}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export SAVE_TRAINING_STATE=0
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv-vlm/bin/python}"
RESULT_DIR="${RESULT_DIR:-$RUN_GROUP/visual_diagnostics_$(date +%Y%m%d_%H%M%S)_$$}"
read -r -a names <<< "${EXPERIMENTS:-pivot_rnn pivot_visual_rnn pivot_history_rnn}"
read -r -a tests <<< "${TESTS:-text-only charxiv_val textvqa}"
extra=()
if [[ "${PROBE_ONLY:-0}" == 1 ]]; then extra+=(--probe-only); fi
"$PYTHON_BIN" tests/run_visual_diagnostics.py --run-group "$RUN_GROUP" --output-dir "$RESULT_DIR" \
  --target-model "${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}" \
  --experiments "${names[@]}" --tests "${tests[@]}" --samples "${EVAL_SAMPLES:-16}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}" --stride "${PROBE_STRIDE:-8}" \
  --seed "${SAMPLE_SEED:-42}" --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}" \
  --state-mode "${STATE_MODE:-per_token}" "${extra[@]}"
echo "Report: $RESULT_DIR/diagnostics.md"
