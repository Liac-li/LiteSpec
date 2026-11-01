# DLite visual KV: context/head attribution and dynamic evidence

Branch: `research/qwen35-vlm-visual-kv`, based on `research/qwen35-vlm-memory`.
All names are persisted in checkpoint `dlite_config.experiment_name`.

## Static context × prediction head

| Experiment | Extra context beyond pivot | Head |
| --- | --- | --- |
| pivot_parallel | none | independent low-rank |
| pivot_rnn | none | original RNN |
| pivot_visual_parallel | image/video placeholder positions in prompt | independent low-rank |
| pivot_visual_rnn | image/video placeholder positions in prompt | original RNN |
| pivot_history_parallel | every committed prefix position except current pivot | independent low-rank |
| pivot_history_rnn | every committed prefix position except current pivot | original RNN |

The history controls are not native DFlash/DSpark implementations. All six
retain the same DLite backbone, multi-layer pivot interface and training loss.
The parallel head does not receive teacher-forced or sampled previous token IDs.
It has a hidden→rank→vocabulary mapping. RNN and parallel parameter counts differ;
exports and evaluation report counts rather than claiming exact parameter matching.

## Source and position convention

Extra context uses one frozen Target language layer (`EVIDENCE_LAYER_ID`, default
middle layer), identically for visual and history modes. No layer averaging,
Top-K filtering, extra memory cap or legacy pivot reader is enabled. Visual start
and end delimiters are excluded. Image token budget is controlled by the existing
processor (default 1024). Text-only examples have zero valid visual evidence.

All experiments use v2 local pivot position 0 and query positions 1..block_size.
Extra evidence keys use identity RoPE (position 0); their original source indices
are retained for causal masks, not treated as text-RoPE positions. This explicit
non-temporal evidence convention prevents mixing global Target RoPE with local
Draft RoPE. It does not introduce new spatial embeddings. Original Target hidden
already contains position-dependent information. Positional alternatives require
separate matched controls; current experiments cannot attribute their differences
solely to information content independently of this convention.

Context layout is `[shared evidence, own pivot slots, own local block]`. Training
masks expose evidence only at positions `< anchor-1`; no other anchor's pivot or
query is visible. Full-history sources are masked before attention, not averaged
across the sequence. CUDA training uses TRITON FlexAttention and shared per-sample
history, avoiding dense replicated per-anchor history tensors. CPU uses dense
SDPA for numerical checks. Inference uses SDPA: FlashAttention2 does not support
these arbitrary masks. Target attention backend remains independently selectable.

Static visual evidence is projected once per layer at prefill. Full-history
projection appends accepted positions only. Candidate query KV is never committed.
No persistent request state is stored on the model object.

## Dynamic evidence

| Experiment | Read bank | Cross-block state |
| --- | --- | --- |
| visual_static_slots | learned fixed queries pool visual evidence into M slots | none |
| visual_pivot_conditioned | current multi-layer pivot mean generates M slot queries | none |
| visual_recurrent | pivot plus recurrent state generates M slot queries | rank-R GRU |

`visual_static_slots` is an additional same-slot-budget control: comparing only
full visual KV with M dynamic slots would confound dynamics with compression.
It is distinct from `pivot_visual_rnn`, which keeps all source visual positions.
Default M=16, R=64; controlled by `VISUAL_SLOTS` and `VISUAL_STATE_RANK`.
The fixed-slot bank and its layer K/V are cached once at inference. Dynamic banks
are rebuilt once per speculative block, and queried in parallel inside that block.
The immutable source visual memory is retained in all three modes. Dynamic slots
are differentiable soft retrievals, not generated factual content or spatial boxes.

GRU is zero-initialized at the end of prefill. Each newly committed token's frozen
Target feature updates it once, in sequence. Training runs a fused GRU over the
teacher-forced answer prefix and gathers state at anchor-1. Inference commits
only accepted candidate features; chunk boundaries have no mathematical effect.
Rejected suffixes never update state. The state is a learned Draft-side controller,
not the Target GDN state. Reader and GRU parameters use AdamW; existing backbone
Muon ownership is unchanged.

Training currently uses teacher-forced Target prefixes, not online speculative
rollout training. This preserves causal state transition semantics but does not
eliminate exposure mismatch. Actual accepted-prefix inference is separately tested.
GPU/FSDP performance, GRU BF16 behavior, compiler specialization and H100 speed
must be checked on the server; CPU tests are not evidence of speedup.

## Unit/tiny-image smoke (no downloaded model)

```bash
PYTHON_BIN="$PWD/.venv-vlm/bin/python" bash tests/smoke_visual_kv.sh
```

## Real-target GPU short training smoke

Use the existing working VLM environment (CUDA training expects torch 2.9.1),
local Qwen3.5-4B and existing regenerated manifests. Do not reinstall its packages.

```bash
export DLITE_WORKSPACE="$HOME/LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm"
cd "$DLITE_WORKSPACE/LiteSpec"
export PYTHON_BIN="$PWD/.venv-vlm/bin/python"
export TARGET_MODEL="$HOME/LMM/lmm_model/Qwen3.5-4B"
export TRAIN_DATA="$DLITE_WORKSPACE/data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl"
export RUN_GROUP="$DLITE_WORKSPACE/runs/visual_kv_smoke_$(date +%Y%m%d_%H%M%S)"
SMOKE_MODE=gpu CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash tests/smoke_visual_kv.sh
```

Smoke intentionally uses a small backbone and two optimizer updates. These
checkpoints must not be mixed with formal ablations. It includes all nine modes.

## Formal static six-group training

```bash
unset MAX_STEPS MAX_SAMPLES DEBUG_RUN NUM_ANCHORS NUM_DRAFT_LAYERS CHS_NUM_LAYERS SEQUENTIAL_RANK
export RUN_GROUP="$DLITE_WORKSPACE/runs/visual_kv_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
MICRO_BATCH_SIZE=2 ACCUMULATION_STEPS=8 EPOCHS=3 BLOCK_SIZE=8 \
bash tests/train_visual_kv.sh
```

Default six static groups, global batch 64. Data validation fixes the eligible set
once and later groups reuse it. Each group is independently trained with the same
seed, backbone initialization and sampling rules; no checkpoints are silently
reused. Use `EXPERIMENTS="pivot_parallel pivot_rnn"` to select named groups.
The launcher preserves upstream loss/optimizer settings. Default save-training-state
is on. Resume with the original configuration and `RESUME=1`; changing debug caps,
world size or data is rejected by the existing runner.

## Dynamic three-group training in the same group

```bash
EXPERIMENTS="visual_static_slots visual_pivot_conditioned visual_recurrent" \
REUSE_VALIDATION_FROM="$RUN_GROUP/pivot_rnn" \
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
MICRO_BATCH_SIZE=2 ACCUMULATION_STEPS=8 EPOCHS=3 \
bash tests/train_visual_kv.sh
```

## Matched decoding evaluation

```bash
EVAL_GPU=0 EVAL_SAMPLES=64 EVAL_MAX_NEW_TOKENS=256 BATCH_SIZES=1 \
TESTS="text-only charxiv_val textvqa" bash tests/eval_visual_kv.sh
```

For dynamic experiments set `EXPERIMENTS="pivot_visual_rnn visual_static_slots
visual_pivot_conditioned visual_recurrent"` (one shell line). Choose a fresh
`--output-dir` on reevaluation. Include Flickr30K/HRBench with the existing dataset
selector names. Default image budget is the training budget; long-image extrapolation
must be explicitly separated. Evaluation reuses exact dataset snapshots and AR
references, tests wrong-image/disabled contexts and resets recurrent state.

Outputs: `comparison.csv`, `summary.json`, `memory_tables.md`, individual request
JSONL including token-match flags and separate prefill/decode/total times. Optional
`PROFILE_DECODE=1` runs separately instrumented timing. Rates are raw observed
rates and are not labeled lossless when AR output differs. Replicated batch is an
offline benchmark, not continuous-batching serving. Wrong-image total time includes
donor prefill; use it only for diagnosis.

## Fixed-prefix visual/state diagnostics (untimed)

```bash
"$PYTHON_BIN" tests/probe_visual_kv.py \
  --target-model "$TARGET_MODEL" --checkpoint "$RUN_GROUP/visual_recurrent/last" \
  --data /path/to/fixed_eval_manifest.jsonl --samples 16 --max-new-tokens 256 \
  --output-dir "$RUN_GROUP/recurrent_probe" --stride 8
```

Reference continuations come from Target-only greedy decoding. Conditions share
exactly the same Target prefix and pivot. Correct/off/wrong-image/reset-state/
wrong-state proposals never enter the Target cache. Wrong-state donors have a
different image but the same question and reference continuation, matched by
number of decoded tokens; source prompt lengths are recorded. Donor visual causal
metadata is rebased because all its evidence is known at prefill.

Reports include per-position tokens, repair/break counts relative to disabled,
first-reference-rejection repair and reference committed length (includes anchor).
Teacher-forced CE uses the fixed reference tokens, not the sampled proposal prefix.
Agreement after the first proposal mismatch is reference-position agreement only,
not actual counterfactual Target verification. The continuous prefix remains a
valid fixed-prefix diagnostic. This probe is not a throughput benchmark.
