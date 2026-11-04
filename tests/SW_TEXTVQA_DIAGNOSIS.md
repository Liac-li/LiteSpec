# TextVQA / 20k DLite window checks

This follow-up intentionally uses only the early 20k DLite parent and TextVQA.
There are no Video-MME, VideoDetailCaption or GSM8K jobs in these entrypoints.
No parameter updates happen during diagnosis.

## Run diagnosis first

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
cd "$DLITE_WORKSPACE/LiteSpec-video-sw-token"
git fetch origin refs/heads/video-sw-token:refs/remotes/origin/video-sw-token
git merge --ff-only origin/video-sw-token
export PYTHON_BIN="$DLITE_WORKSPACE/LiteSpec/.venv-vlm/bin/python"
export RESULT_DIR="$DLITE_WORKSPACE/runs/sw_textvqa_diagnosis_$(date +%Y%m%d_%H%M%S)_$$"
EVAL_GPU=7 EVAL_SAMPLES=16 EVAL_MAX_NEW_TOKENS=256 \
PROBE_STRIDE=8 PROBES_PER_SAMPLE=16 bash tests/diagnose_sw_token.sh
cat "$RESULT_DIR/diagnostics.md"
```

The parent defaults to
`runs/vlm_three_3ep_20261021_045040_2983971/dlite/last`. The actual step directory
is resolved and its config hash/weight inventory recorded. This is a model
trained on approximately 20k FineVision candidates, **not** 20k TextVQA samples.
The diagnostic uses 16 TextVQA examples by default, seed 42 and image cap 1024.
It reuses the existing TextVQA evaluation adapter and freezes `inputs.jsonl`.

One Target AR trajectory supplies each prefix and next-token reference. Only
Target-generated tokens enter its cache; interventions are evaluated afterwards.
The same known anchor, preceding hidden features, and future AR reference are
shared by all four conditions:

| Condition | Old position | Pivot | Rotary coordinates |
|---|---|---|---|
| k1 | absent | present | Original K=1 |
| k2 | present | present | Native K=2 |
| k2_pivot_only | masked | present | Native K=2 |
| k2_pivot_only_origin0 | masked | present | ALL coordinates shifted by -1 |

The last control preserves pivot-query relative positions and restores their
original absolute coordinates. Masked old keys receive no attention; their
negative rotary coordinate is never visible. This distinguishes extra-key
interference from absolute-phase rounding or positional/mask bugs. No layer
averaging, feature compression or additional visual KV bank is introduced.

### Read the report

- Alignment: hidden/logit max/mean deltas, tolerance checks and exact first greedy
  proposal agreement vs K=1. Defaults `atol=0.05`, `rtol=0.01` accommodate BF16;
  use CLI `--alignment-atol`/`--alignment-rtol` to choose stricter numeric checks.
  Both masked controls must pass, with nonempty probes. Tolerances are saved.
- Prediction: first candidate hit (excludes known anchor), greedy proposal
  agreement, first repair/break and fixed-prefix accepted length (includes anchor).
  The final short tail is capped by available AR tokens; this is not closed-loop
  acceptance or a speed benchmark. CE and logit deltas share identical
  teacher-forced Target previous tokens in the RNN head.
- Attention: actual post-normalization/RoPE Q/K, grouped-query key expansion and
  FP32 softmax over **old + pivot + block keys**, with the actual mask. The original
  SDPA forward is left intact. Per-layer, per-raw-query-index mass is averaged
  across heads and then prefixes. Per-valid-slot values account for group size.
  Query index is the raw Draft query index, not an asserted output-token mapping.
  Masked old mass should be zero; each row's three masses should sum to one.
  Attention mass is descriptive, not proof of usefulness.

Outputs: `settings.json`, `inputs.jsonl`, `ar_sequences.jsonl`, `probes.jsonl`,
`attention.jsonl`, `attention.csv`, `summary.json`, `diagnostics.md`. Per-probe
rows distinguish first-round from later AR-prefix samples. Later AR prefixes are
shared interventions, not model-dependent closed-loop boundaries. Prefixes whose
known anchor is EOS or which have no candidate before the cap are excluded.
A failed alignment still writes the report, then exits with status 2.

## Only after PASS: train a matched pair

```bash
export DIAGNOSTICS_DIR="$RESULT_DIR"
export RUN_GROUP="$DLITE_WORKSPACE/runs/sw_pair_20k_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
MICRO_BATCH_SIZE=2 ACCUMULATION_STEPS=8 EPOCHS=1 EVAL_GPU=7 \
bash tests/train_sw_token_pair.sh
```

The launcher refuses missing/failed alignment, changed parent checkpoints and
changed diagnostic sample selections. K=1 and K=2 both load the same diagnosed
parent, each with fresh optimizer/schedule and identical additional epochs/data.
K=1 is a **continued-training control**, not the unmodified parent. Parent block
size/head/layer selection are retained. All K use chunked SDPA training.

Default training corpus is
`data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl` (override
`TRAIN_DATA` explicitly if required). This validated 20k corpus need not equal the
old parent's original candidate set. Default 4 GPUs, micro batch 2, accumulation 8
means global batch 64. One additional epoch is the quick check; use `EPOCHS=3`
for three. No training on TextVQA evaluation examples occurs.

The launcher forces `SAVE_TRAINING_STATE=0` and clears debug sample/step caps.
Only weights/config and ordinary training metrics are exported. It does not
write `training_state_rank_*`. Existing data eligibility validation is still
performed; `REUSE_VALIDATION_FROM` may supply a compatible prior validation.

After both runs, `EVAL_AFTER_TRAIN=1` (default) benchmarks parent, trained K=1 and
trained K=2 on the **exact frozen diagnostic TextVQA prompts**, B=1, per_token,
with the diagnostic image/output budgets. GPU selection defaults to 7. Results
are under `$RUN_GROUP/textvqa_evaluation`; raw speed ratios are reported alongside
AR token match. Set `EVAL_AFTER_TRAIN=0` to omit this final evaluation.
