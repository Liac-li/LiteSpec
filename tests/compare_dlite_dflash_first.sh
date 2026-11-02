#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export SAVE_TRAINING_STATE=0
PAIR_RUN="${PAIR_RUN:-$DLITE_WORKSPACE/runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049}"
RESULT_DIR="${RESULT_DIR:-$PAIR_RUN/first_candidate_pair_$(date +%Y%m%d_%H%M%S)_$$}"
PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
read -r -a tests <<< "${TESTS:-text-only charxiv_val textvqa}"
"$PYTHON_BIN" tests/compare_dlite_dflash_first.py --pair-run "$PAIR_RUN" --output-dir "$RESULT_DIR" \
  --target-model "${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}" \
  --tests "${tests[@]}" --samples "${EVAL_SAMPLES:-64}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}" \
  --seed "${SAMPLE_SEED:-42}" --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}" \
  --state-mode "${STATE_MODE:-per_token}"
echo "Report: $RESULT_DIR/first_candidate_report.md"
