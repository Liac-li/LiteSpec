#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-$DLITE_WORKSPACE/runs/vlm_three_3ep_20261021_045040_2983971/dlite/last}"
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/video_sw_token_$(date +%Y%m%d_%H%M%S)_$$}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
export IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
export EPOCHS="${EPOCHS:-1}"
export SW_TRAIN_CHUNK_BLOCKS="${SW_TRAIN_CHUNK_BLOCKS:-16}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export SAVE_TRAINING_STATE=0
export ALGORITHM=dlite
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" tests/check_sw_token_source.py
read -r -a windows <<< "${WINDOW_SIZES:-1 2 5 10}"
[[ -s "$TRAIN_DATA" ]] || { echo "Missing training data: $TRAIN_DATA" >&2; exit 1; }
# Validate everything and pin last to one fixed parent BEFORE any training.
INIT_CHECKPOINT="$($PYTHON_BIN - <<'PY'
import os
from pathlib import Path
from litespec.vlm.sw_token import warm_start_config
p=Path(os.environ['INIT_CHECKPOINT']).expanduser().resolve(strict=True)
cfg={}; warm_start_config(cfg,p)
print(p)
PY
)"
export INIT_CHECKPOINT
[[ ! -e "$RUN_GROUP" ]] || { echo "Choose a fresh RUN_GROUP: $RUN_GROUP" >&2; exit 1; }
"$PYTHON_BIN" - "${windows[@]}" <<'PY'
import sys
values=[int(v) for v in sys.argv[1:]]
if not values or min(values)<1 or len(set(values))!=len(values): raise ValueError('unique positive windows required')
PY
mkdir -p "$RUN_GROUP"
reuse="${REUSE_VALIDATION_FROM:-}"
for k in "${windows[@]}"; do
    export OUTPUT_DIR="$RUN_GROUP/k$k" CONFIG="$RUN_GROUP/k$k/config.yaml"
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" - "$k" <<'PY'
import os,sys,yaml
from pathlib import Path
from litespec.vlm.sw_token import warm_start_config
cfg=yaml.safe_load(Path('scripts/config/vlm/sft_student_aligned.yaml').read_text())
warm_start_config(cfg,os.environ['INIT_CHECKPOINT'])
cfg.update(sw_chunked_training=True,context_window_size=int(sys.argv[1]), init_checkpoint=os.environ['INIT_CHECKPOINT'],
    image_max_tokens=int(os.environ['IMAGE_MAX_TOKENS']),
    sw_train_chunk_blocks=int(os.environ['SW_TRAIN_CHUNK_BLOCKS']),flex_attention_backend='TRITON',
    num_epochs=int(os.environ['EPOCHS']),batch_size=int(os.environ['MICRO_BATCH_SIZE']),
    accumulation_steps=int(os.environ['ACCUMULATION_STEPS']))
Path(os.environ['CONFIG']).write_text(yaml.safe_dump(cfg,sort_keys=False))
print(f"K={sys.argv[1]}: same parent={cfg['init_checkpoint']}; extra epochs={cfg['num_epochs']}; block={cfg['block_size']}",flush=True)
PY
    if [[ -n "$reuse" ]]; then export REUSE_VALIDATION_FROM="$reuse"; else unset REUSE_VALIDATION_FROM; fi
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
    reuse="$OUTPUT_DIR"
done
echo "Training complete: $RUN_GROUP (weights only; fresh optimizers; no training_state_rank files)"
