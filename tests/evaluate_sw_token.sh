#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
RESULT_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/video_sw_scan_$(date +%Y%m%d_%H%M%S)_$$}"
read -r -a windows <<< "${WINDOW_SIZES:-1 2 5 10}"
read -r -a tests <<< "${TESTS:-Video-MME VideoDetailCaption textvqa text-only}"
read -r -a batches <<< "${BATCH_SIZES:-1}"
extra=()
if [[ -n "${SW_RUN_GROUP:-}" ]]; then
    extra+=(--trained-window)
    for k in "${windows[@]}"; do extra+=(--checkpoint "trained_k$k" "$SW_RUN_GROUP/k$k/last"); done
else
    if [[ -n "${DLITE_20K:-}" || -n "${DLITE_100K:-}" ]]; then
        [[ -z "${DLITE_20K:-}" ]] || extra+=(--checkpoint dlite_20k "$DLITE_20K")
        [[ -z "${DLITE_100K:-}" ]] || extra+=(--checkpoint dlite_100k "$DLITE_100K")
    fi
fi
"$PYTHON_BIN" tests/evaluate_sw_token.py --workspace "$DLITE_WORKSPACE" --output-dir "$RESULT_DIR" \
  --target-model "${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}" --windows "${windows[@]}" \
  --tests "${tests[@]}" --samples "${EVAL_SAMPLES:-16}" --batch-sizes "${batches[@]}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}" --seed "${SAMPLE_SEED:-42}" \
  --frames "${VIDEO_FRAMES:-16}" --max-pixels-per-frame "${VIDEO_MAX_PIXELS:-200704}" \
  --video-data-root "${VIDEO_DATA_ROOT:-$HOME/LMM/liyt/datasets/eval_bench}" \
  --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}" "${extra[@]}" "$@"
