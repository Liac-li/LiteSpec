#!/usr/bin/env bash
# Three matched Reader-only adaptations. Never starts training from scratch.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO_ROOT")}"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv-vlm/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}"
export ORIGINAL_DLITE="${ORIGINAL_DLITE:-$HOME/LMM/lmm_model/Qwen3.5-4B-DLite}"
export TRAIN_DATA="${TRAIN_DATA:-$DLITE_WORKSPACE/data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export SAVE_TRAINING_STATE=0
export EPOCHS="${EPOCHS:-3}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LITESPEC_EAGER_BLOCK_MASK=1
unset MAX_STEPS MAX_SAMPLES
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
    export MAX_STEPS=2 MAX_SAMPLES=32 EPOCHS=1 ACCUMULATION_STEPS=1
fi
cd "$REPO_ROOT"
[[ -d "$ORIGINAL_DLITE" && -s "$TRAIN_DATA" ]] || { echo 'Missing ORIGINAL_DLITE or TRAIN_DATA' >&2; exit 1; }
"$PYTHON_BIN" - "$TRAIN_DATA" <<'PY'
import os,sys
count=sum(bool(line.strip()) for line in open(sys.argv[1]))
if os.environ.get('SMOKE_TEST')!='1' and count!=20000:
    raise SystemExit(f'Expected the existing 20000-row training manifest, got {count}')
PY
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/frozen_gdn_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "$RUN_GROUP" ]] || { echo 'Choose a fresh RUN_GROUP' >&2; exit 1; }
mkdir -p "$RUN_GROUP"
"$PYTHON_BIN" tests/check_frozen_gdn_cuda.py --target "$TARGET_MODEL" --draft "$ORIGINAL_DLITE" \
    --data "$TRAIN_DATA" --output "$RUN_GROUP/preflight.json"
for arm in ${EXPERIMENTS:-frozen_reader_legacy_loss frozen_reader_ready_loss frozen_reader_content_only}; do
    export ARM="$arm" OUTPUT_DIR="$RUN_GROUP/$arm"
    mkdir -p "$OUTPUT_DIR"
    export CONFIG="$OUTPUT_DIR/config.yaml"
    "$PYTHON_BIN" - <<'PY'
import os,yaml
from pathlib import Path
arm=os.environ['ARM']
if arm not in {'frozen_reader_legacy_loss','frozen_reader_ready_loss','frozen_reader_content_only'}:
    raise SystemExit(f'Unknown experiment: {arm}')
cfg=yaml.safe_load(Path('scripts/config/vlm/frozen_reader.yaml').read_text())
cfg.update(pretrained_draft=os.environ['ORIGINAL_DLITE'],target_model=os.environ['TARGET_MODEL'],
           num_epochs=int(os.environ['EPOCHS']),batch_size=int(os.environ['MICRO_BATCH_SIZE']),
           accumulation_steps=int(os.environ['ACCUMULATION_STEPS']))
if arm=='frozen_reader_legacy_loss':cfg.update(final_ce_weight=.1,l1_loss_weight=1.)
if arm=='frozen_reader_content_only':cfg['gdn_feature_mode']='content_only'
Path(os.environ['CONFIG']).write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
    bash scripts/dlite/run_training_vlm.sh 2>&1 | tee "$OUTPUT_DIR/train.log"
done
echo "Completed Reader-only adaptations: $RUN_GROUP"
