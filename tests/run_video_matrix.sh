#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv-vlm/bin/python}"
export RESULT_DIR="${RESULT_DIR:-$DLITE_WORKSPACE/runs/video_matrix_$(date +%Y%m%d_%H%M%S)}"
read -r -a datasets <<< "${TESTS:-VideoDetailCaption Video-MME}"
read -r -a models <<< "${MODELS:-dlite_original dlite_vlm_100k dflash_vlm_100k}"
read -r -a batches <<< "${BATCH_SIZES:-1 16}"
args=(--workspace "$DLITE_WORKSPACE" --output-dir "$RESULT_DIR"
      --datasets "${datasets[@]}" --models "${models[@]}" --batch-sizes "${batches[@]}"
      --samples "${EVAL_SAMPLES:-16}" --frames "${VIDEO_FRAMES:-16}"
      --max-pixels-per-frame "${VIDEO_MAX_PIXELS:-200704}"
      --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-256}" --seed "${SAMPLE_SEED:-42}"
      --warmup "${WARMUP:-1}")
[[ -z "${TARGET_MODEL:-}" ]] || args+=(--target-model "$TARGET_MODEL")
[[ -z "${ORIGINAL_DLITE:-}" ]] || args+=(--original-dlite "$ORIGINAL_DLITE")
[[ -z "${DLITE_VLM:-}" ]] || args+=(--dlite-vlm "$DLITE_VLM")
[[ -z "${DFLASH_VLM:-}" ]] || args+=(--dflash-vlm "$DFLASH_VLM")
[[ -z "${VIDEO_DATA_ROOT:-}" ]] || args+=(--data-root "$VIDEO_DATA_ROOT")
[[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
[[ "${INSPECT_ONLY:-0}" != 1 ]] || args+=(--inspect-only)
exec "$PYTHON_BIN" tests/benchmark_video.py "${args[@]}" "$@"
