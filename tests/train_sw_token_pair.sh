#!/usr/bin/env bash
# Continue K=1/K=2 from the exact diagnosed 20k parent, only after alignment PASS.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DLITE_WORKSPACE="${DLITE_WORKSPACE:-$(dirname -- "$REPO")}"
export PYTHON_BIN="${PYTHON_BIN:-$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
: "${DIAGNOSTICS_DIR:?Set DIAGNOSTICS_DIR to a completed TextVQA probe directory}"
export DIAGNOSTICS_DIR
"$PYTHON_BIN" tests/check_sw_token_source.py
INIT_CHECKPOINT="$($PYTHON_BIN - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['DIAGNOSTICS_DIR']).expanduser().resolve(strict=True)
r=json.loads((root/'summary.json').read_text())
if r.get('schema')!='sw_textvqa_probe_v1' or r.get('alignment_passed') is not True:
    raise ValueError('Alignment must PASS before training; inspect diagnostics.md')
path=Path(r['checkpoint']['path']).resolve(strict=True)
actual=dict(path=str(path), config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
    weights=[dict(name=p.name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
             for p in sorted(path.iterdir()) if p.suffix in ('.safetensors','.bin')])
if actual!=r['checkpoint']:
    raise ValueError('Parent checkpoint changed since diagnosis')
if hashlib.sha256((root/'inputs.jsonl').read_bytes()).hexdigest()!=r['selection_sha256']:
    raise ValueError('Diagnostic selection changed')
print(path)
PY
)"
export INIT_CHECKPOINT
TARGET_MODEL="$("$PYTHON_BIN" - <<'PYTHON'
import json,os
from pathlib import Path
r=json.loads((Path(os.environ['DIAGNOSTICS_DIR'])/'settings.json').read_text())
print(r['arguments']['target_model'])
PYTHON
)"
export TARGET_MODEL
export WINDOW_SIZES="1 2"
export RUN_GROUP="${RUN_GROUP:-$DLITE_WORKSPACE/runs/sw_pair_20k_$(date +%Y%m%d_%H%M%S)_$$}"
export EPOCHS="${EPOCHS:-1}"
export SAVE_TRAINING_STATE=0
unset MAX_STEPS MAX_SAMPLES CONFIG OUTPUT_DIR ALGORITHM
bash tests/train_sw_token.sh
if [[ "${EVAL_AFTER_TRAIN:-1}" == 1 ]]; then
    readarray -t protocol < <("$PYTHON_BIN" - <<'PY'
import json,os
from pathlib import Path
r=json.loads((Path(os.environ['DIAGNOSTICS_DIR'])/'settings.json').read_text())['arguments']
for key in ('target_model','samples','max_new_tokens','image_max_tokens','seed'): print(r[key])
PY
)
    pixels="$($PYTHON_BIN - "${protocol[0]}" "${protocol[3]}" <<'PY'
import json,sys
from pathlib import Path
v=json.loads((Path(sys.argv[1])/'config.json').read_text())['vision_config']
print(int(sys.argv[2])*(v['patch_size']*v['spatial_merge_size'])**2)
PY
)"
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}" "$PYTHON_BIN" tests/benchmark_vlm_static_batch.py \
      --target-model "${protocol[0]}" --data "$DIAGNOSTICS_DIR/inputs.jsonl" \
      --dataset-name textvqa --max-samples "${protocol[1]}" --seed "${protocol[4]}" \
      --max-new-tokens "${protocol[2]}" --image-max-pixels "$pixels" \
      --checkpoint parent_20k "$INIT_CHECKPOINT" \
      --checkpoint trained_k1 "$RUN_GROUP/k1/last" --checkpoint trained_k2 "$RUN_GROUP/k2/last" \
      --batch-sizes 1 --state-mode per_token --attention-backend sdpa --cpu-inputs \
      --output-dir "$RUN_GROUP/textvqa_evaluation"
fi
