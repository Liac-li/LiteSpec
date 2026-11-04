#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
: "${RUN_GROUP:?Set RUN_GROUP to the completed frozen_gdn run}"
args=()
for arm in ${EXPERIMENTS:-frozen_reader_legacy_loss frozen_reader_ready_loss frozen_reader_content_only}; do
    args+=(--checkpoint "$arm" "$RUN_GROUP/$arm/last")
done
export MODELS=dlite_original
export RESULT_DIR="${RESULT_DIR:-$RUN_GROUP/video_evaluation_$(date +%Y%m%d_%H%M%S)}"
exec bash tests/run_video_matrix.sh "${args[@]}" "$@"
