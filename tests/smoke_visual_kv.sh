#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv-vlm/bin/python}"
if [[ "${SMOKE_MODE:-unit}" == unit ]]; then
    exec "$PYTHON_BIN" -m pytest tests/test_visual_kv.py tests/test_visual_kv_runner.py -q
fi
export DEBUG_RUN=1 MAX_STEPS="${MAX_STEPS:-2}" MAX_SAMPLES="${MAX_SAMPLES:-16}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}" MICRO_BATCH_SIZE=1 ACCUMULATION_STEPS=1
export NUM_ANCHORS=4 NUM_DRAFT_LAYERS=1 CHS_NUM_LAYERS=4 SEQUENTIAL_RANK=64 EPOCHS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXPERIMENTS="${EXPERIMENTS:-pivot_parallel pivot_rnn pivot_visual_parallel pivot_visual_rnn pivot_history_parallel pivot_history_rnn visual_static_slots visual_pivot_conditioned visual_recurrent}"
export REPORT_TO=none
exec bash tests/train_visual_kv.sh
