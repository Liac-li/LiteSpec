#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv-vlm/bin/python}"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
export EPOCHS="${EPOCHS:-3}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
export IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-1024}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export SAVE_TRAINING_STATE="${SAVE_TRAINING_STATE:-1}"
export LITESPEC_EAGER_BLOCK_MASK="${LITESPEC_EAGER_BLOCK_MASK:-1}"
export EXPERIMENTS="${EXPERIMENTS:-pivot_parallel pivot_rnn pivot_visual_parallel pivot_visual_rnn pivot_history_parallel pivot_history_rnn}"
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/visual_kv_$(date +%Y%m%d_%H%M%S)_$$}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${DEBUG_RUN:-0}" != 1 && ( -n "${MAX_STEPS:-}" || -n "${MAX_SAMPLES:-}" ) ]]; then
    echo 'Debug limits require DEBUG_RUN=1.' >&2; exit 2
fi
[[ -s "$TRAIN_DATA" ]] || { echo "Missing data: $TRAIN_DATA" >&2; exit 1; }
# Validate ALL names before creating output or starting a long run.
"$PYTHON_BIN" - <<'PY'
import os
from litespec.modeling.draft.visual_kv import experiment_settings
names=os.environ['EXPERIMENTS'].split()
if len(names)!=len(set(names)): raise ValueError('duplicate experiments')
for name in names: experiment_settings(name)
PY
mkdir -p "$RUN_GROUP"
reuse="${REUSE_VALIDATION_FROM:-}"
for name in $EXPERIMENTS; do
    export OUTPUT_DIR="$RUN_GROUP/$name" CONFIG="$RUN_GROUP/$name/config.yaml" ALGORITHM=dlite
    if [[ -e "$OUTPUT_DIR/last" && "${RESUME:-0}" != 1 ]]; then
        echo "Checkpoint exists: $OUTPUT_DIR; use RESUME=1 or a new group." >&2; exit 2
    fi
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" - "$CONFIG" "$name" <<'PY'
import os,sys,yaml
from pathlib import Path
from litespec.modeling.draft.visual_kv import experiment_settings
cfg=yaml.safe_load(Path('scripts/config/vlm/sft_student_aligned.yaml').read_text())
cfg.update(experiment_settings(sys.argv[2]), memory_mode='none', flex_attention_backend='TRITON')
for key,env in [('num_epochs','EPOCHS'),('batch_size','MICRO_BATCH_SIZE'),('accumulation_steps','ACCUMULATION_STEPS'),('block_size','BLOCK_SIZE'),('image_max_tokens','IMAGE_MAX_TOKENS'),('seed','TRAIN_SEED'),('num_anchors','NUM_ANCHORS'),('num_draft_layers','NUM_DRAFT_LAYERS'),('chs_num_layers','CHS_NUM_LAYERS'),('sequential_rank','SEQUENTIAL_RANK'),('evidence_layer_id','EVIDENCE_LAYER_ID'),('visual_slots','VISUAL_SLOTS'),('visual_state_rank','VISUAL_STATE_RANK')]:
    if env in os.environ: cfg[key]=int(os.environ[env])
cfg.setdefault('visual_slots',16);cfg.setdefault('visual_state_rank',64)
Path(sys.argv[1]).write_text(yaml.safe_dump(cfg,sort_keys=False))
print(f"{sys.argv[2]}: global batch={cfg['batch_size']*cfg['accumulation_steps']*int(os.environ['NPROC_PER_NODE'])}",flush=True)
PY
    if [[ -n "$reuse" ]]; then export REUSE_VALIDATION_FROM="$reuse"; else unset REUSE_VALIDATION_FROM; fi
    extra=()
    if [[ "${RESUME:-0}" == 1 && -e "$OUTPUT_DIR/last" ]]; then extra+=(--resume "$OUTPUT_DIR/last"); fi
    bash scripts/dlite/run_training_vlm.sh "${extra[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
    reuse="$OUTPUT_DIR"
done
if [[ "${RUN_EVAL:-0}" == 1 ]]; then bash tests/eval_visual_kv.sh; fi
echo "Completed: $RUN_GROUP"
